
# Copyright (C) 2023 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# AQuaSurF Software in commercial settings.
#
# END COPYRIGHT
"""
Database class for storing activation function data.
"""

import gc
import sqlite3

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Needed for evaluating data stored as strings
from numpy import inf # pylint: disable=unused-import
from numpy import nan # pylint: disable=unused-import
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm
from umap import UMAP

from aquasurf.activation import ActivationFunction
from aquasurf.activation import BINARY_FUNCTIONS
from aquasurf.activation import N_ARY_FUNCTIONS
from aquasurf.activation import UNARY_FUNCTIONS
from aquasurf.fisher import FIM


class ActivationFunctionDatabase: # pylint: disable=too-many-instance-attributes
    """
    This is the base class for activation function databases.  Subclasses must
    override the build_model method to build a model with the given activation,
    as well as define self.samples, self.labels, self.loss, and self.weights_per_layer
    to be used for computing the eigenvalues of the Fisher.
    """

    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        # Update the sqlite timeout to 1 hour
        self.cursor.execute('PRAGMA busy_timeout = 3600000')
        self.conn.commit()

        # Increase the max database size
        self.cursor.execute('PRAGMA max_page_count = 2147483646')
        self.conn.commit()

        # Defined in subclasses
        self.samples = None
        self.labels = None
        self.loss = None
        self.weights_per_layer = None
        self.baseline_fns = None

        self.umap_fn_outputs = None
        self.umap_fisher_eigs = None
        self.umap_union = None

        self.regression_model = KNeighborsRegressor(n_neighbors=3)

        np.random.seed(0)
        self.xvals = np.clip(np.random.normal(size=1000), -5, 5)

        self.database_columns = [
            'fn_name',
            'fn_outputs',
            'fisher_eigs',
            'status',
            'train_acc',
            'train_loss',
            'val_acc',
            'val_loss',
            'test_acc',
            'test_loss',
            'runtime',
            'WandB_config',
            'WandB_summary',
        ]

        # create the table if it doesn't exist
        self.cursor.execute(
            'CREATE TABLE IF NOT EXISTS activation_functions ({})'.format(
                ', '.join(self.database_columns)
            )
        )
        self.conn.commit()


    def build_model(self, fn_name):
        """
        Build a model with the given activation function.
        """
        # Must be overridden by subclasses.
        raise NotImplementedError


    def clean(self):
        """
        Keep fn_name, fn_outputs, and fisher_eigs, but reset status to "not evaluated"
        and remove all other values.
        """
        self.cursor.execute(
            'UPDATE activation_functions SET status = "not evaluated", ' \
                'train_acc = NULL, train_loss = NULL, val_acc = NULL, val_loss = NULL, ' \
                'test_acc = NULL, test_loss = NULL, runtime = NULL, WandB_config = NULL, ' \
                'WandB_summary = NULL'
        )
        self.conn.commit()


    def soft_clean(self):
        """
        Reset the status for all functions for which we don't have a val_acc, but keep results
        for runs that did finish.
        """
        self.cursor.execute(
        'UPDATE activation_functions SET status = "not evaluated", ' \
            'train_acc = NULL, train_loss = NULL, val_acc = NULL, val_loss = NULL, ' \
            'test_acc = NULL, test_loss = NULL, runtime = NULL, WandB_config = NULL, ' \
            'WandB_summary = NULL ' \
                'WHERE val_acc IS NULL'
        )
        self.conn.commit()


    def get_fn_names_from_schema(self, schema):
        """
        Get the names of the activation functions from the given schema.
        The schema should be the name of a potential ActivationFunction,
        except with generic operator types unary, binary, or n-ary.
        For example, the schema binary(unary(x),unary(x)) would return
        add(zero(x),zero(x)), add(zero(x),one(x)), ..., min(hard_sigmoid(x),hard_sigmoid(x)).
        """
        fn_names = []
        num_nary = schema.count('n-ary')
        num_binary = schema.count('binary')
        num_unary = schema.count('unary')
        op_names_with_repeats = [list(N_ARY_FUNCTIONS)] * num_nary + \
                                [list(BINARY_FUNCTIONS)] * num_binary + \
                                [list(UNARY_FUNCTIONS)] * num_unary
        for op_names in product(*op_names_with_repeats):
            fn_name = schema
            for nary_op in op_names[:num_nary]:
                fn_name = fn_name.replace('n-ary', nary_op, 1)
            for binary_op in op_names[num_nary:num_nary+num_binary]:
                fn_name = fn_name.replace('binary', binary_op, 1)
            for unary_op in op_names[-num_unary:]:
                fn_name = fn_name.replace('unary', unary_op, 1)
            fn_names.append(fn_name)
        return fn_names


    def fn_outputs(self, fn_name):
        """
        Return the outputs of the given activation function.
        """
        return list(ActivationFunction(fn_name)(self.xvals).numpy())


    def populate_database(self, schema=None, fn_names_list=None):
        """
        Populate the database with functions from the given schema or list, skipping
        any that are already in the database.  Set the status to "not evaluated."
        """
        if schema is not None:
            fn_names = self.get_fn_names_from_schema(schema)
        else:
            fn_names = fn_names_list

        self.cursor.executemany(
            'INSERT OR IGNORE INTO activation_functions (fn_name, fn_outputs, status) ' \
                'VALUES (?, ?, ?)',
            [(fn_name, str(self.fn_outputs(fn_name)), 'not evaluated')
                for fn_name in fn_names],
        )
        self.conn.commit()


    def print_top_n(self, top_n=None):
        """
        Print the top n elements in the database.
        If n is None, print all elements.
        """
        self.cursor.execute('SELECT * FROM activation_functions')
        for row in self.cursor.fetchall()[:top_n]:
            print(row)


    def summary(self):
        """
        Print summary statistics about the database.
        """
        num_fns = self.cursor.execute('SELECT COUNT(*) FROM activation_functions').fetchone()[0]
        num_unique_fns = self.cursor.execute(
            'SELECT COUNT(DISTINCT fn_outputs) FROM activation_functions').fetchone()[0]
        num_unique_with_eigs = self.cursor.execute(
            'SELECT COUNT(DISTINCT fn_outputs) FROM activation_functions WHERE ' \
                'fisher_eigs IS NOT NULL').fetchone()[0]
        num_evaluated = self.cursor.execute(
            'SELECT COUNT(*) FROM activation_functions WHERE status = "done"').fetchone()[0]
        num_running = self.cursor.execute(
            'SELECT COUNT(*) FROM activation_functions WHERE status = "running"').fetchone()[0]
        best_so_far = self.cursor.execute(
            'SELECT fn_name, val_acc FROM activation_functions '\
                'ORDER BY val_acc DESC LIMIT 1').fetchone()

        print(f'Number of activation functions: {num_fns}')
        print(f'Number of unique activation functions: {num_unique_fns}')
        print(f'Number of unique functions with eigenvalues calculated: {num_unique_with_eigs}')
        print(f'Number of evaluated activation functions: {num_evaluated}')
        print(f'Number of running activation functions: {num_running}')
        print(f'Best so far: {best_so_far[0]} with validation accuracy {best_so_far[1]}')


    def get_unique_fn_names(self, condition=None):
        """
        Return a list of function names from the database that have unique outputs,
        optionally filtered by the given condition.
        """
        unique_fn_names = self.cursor.execute(
            'SELECT MAX(fn_name) FROM activation_functions '\
                f'{"WHERE " + condition if condition else ""} '\
                'GROUP BY fn_outputs'
        ).fetchall()
        return [row[0] for row in unique_fn_names]


    def fit_umap_models(self):
        """
        Fit UMAP models to the function outputs and Fisher eigenvalues.
        """
        data = self.cursor.execute(
            'SELECT MAX(fn_name), fn_outputs, fisher_eigs FROM activation_functions '\
                'WHERE fn_outputs IS NOT NULL AND fisher_eigs IS NOT NULL '\
                'GROUP BY fn_outputs'
        ).fetchall()

        # Filter out cases with undefined Fisher eigenvalues or nan/inf values.
        fn_names = []
        fn_outputs = []
        fisher_eigs = []
        for row in data:
            fn_name = row[0]
            outputs = eval(row[1]) # pylint: disable=eval-used
            eigs = eval(row[2]) # pylint: disable=eval-used
            # Really we would check len(eigs) > 0, but for some reason some ResNet-56 entries
            # have just three values, and a few MobileViTv2-0.5 entries have 339 values,
            # and we don't want those.
            # print(len(eigs))
            if len(eigs) > 141000 and all(np.isfinite(eigs)) and all(np.isfinite(outputs)):
                fn_names.append(fn_name)
                fn_outputs.append(outputs)
                fisher_eigs.append(eigs)

        fn_outputs = np.array(fn_outputs)
        fisher_eigs = np.stack(fisher_eigs)

        self.umap_fn_outputs = UMAP(metric='euclidean').fit(fn_outputs)
        self.umap_fisher_eigs = UMAP(metric='manhattan', n_neighbors=3).fit(fisher_eigs)
        self.umap_union = self.umap_fn_outputs + self.umap_fisher_eigs
        return fn_names, self.umap_union.embedding_


    def suggest_fn(self):
        """
        Suggest a function to try next based on predicted accuracy.
        """
        # If there are any baseline functions that haven't been
        # evaluated (or aren't running), try one of those first.
        self.cursor.execute(
            'SELECT fn_name FROM activation_functions '\
                'WHERE status = "not evaluated" AND fn_name IN ({}) '.format(
                ','.join('?' * len(self.baseline_fns))
            ),
            self.baseline_fns
        )
        unevaluated_baselines = [row[0] for row in self.cursor.fetchall()]
        if len(unevaluated_baselines) > 0:
            return np.random.choice(unevaluated_baselines)

        # Otherwise, try to predict the best function to try next.
        fn_names, embeddings = self.fit_umap_models()

        # Functions in the training set are the ones we already have an accuracy for.
        train_data = self.cursor.execute(
            'SELECT fn_name, val_acc FROM activation_functions WHERE fn_name IN ({}) '\
                'AND val_acc IS NOT NULL'.format(
                ','.join('?' * len(fn_names))
            ),
            fn_names
        ).fetchall()
        train_fn_names = [row[0] for row in train_data]
        train_val_accs = [row[1] for row in train_data]

        # Functions in the test set are the ones we don't have an accuracy for,
        # and the ones we aren't currently evaluating.
        test_fn_names = self.cursor.execute(
            'SELECT fn_name FROM activation_functions WHERE fn_name IN ({}) '\
                'AND val_acc IS NULL AND status != "running"'.format(
                ','.join('?' * len(fn_names))
            ),
            fn_names
        ).fetchall()
        test_fn_names = [row[0] for row in test_fn_names]

        train_embeddings = embeddings[[fn_names.index(fn_name) for fn_name in train_fn_names]]
        test_embeddings = embeddings[[fn_names.index(fn_name) for fn_name in test_fn_names]]

        try:
            self.regression_model.fit(train_embeddings, train_val_accs)
            predicted_accs = self.regression_model.predict(test_embeddings)
            best_fn_name = test_fn_names[np.argmax(predicted_accs)]
        except ValueError:
            # If there are not enough training samples, return a random function.
            best_fn_name = np.random.choice(test_fn_names)
        return best_fn_name


    def update_status(self, fn_name, status):
        """
        Update the status of the given function.
        """
        self.cursor.execute(
            'UPDATE activation_functions SET status = ? WHERE fn_name = ?',
            (status, fn_name)
        )
        self.conn.commit()


    def update_status_all_equivalent_fns(self, fn_name, status):
        """
        Update the status of all functions equivalent to the given function.
        """
        fn_outputs = self.cursor.execute(
            'SELECT fn_outputs FROM activation_functions WHERE fn_name = ?',
            (fn_name,)
        ).fetchone()[0]
        self.cursor.execute(
            'UPDATE activation_functions SET status = ? WHERE fn_outputs = ?',
            (status, fn_outputs)
        )
        self.conn.commit()


    def update_result(self, fn_name, result_name, result_value):
        """
        Update the result of the given function.
        """
        self.cursor.execute(
            'UPDATE activation_functions SET {} = ? WHERE fn_name = ?'.format(result_name),
            (result_value, fn_name)
        )
        self.conn.commit()


    def update_for_all_equivalent_fns(self, fn_name, result_name, result_value):
        """
        Update the result of the given function and all equivalent functions.
        """
        fn_output = self.cursor.execute(
            'SELECT fn_outputs FROM activation_functions WHERE fn_name = ?',
            (fn_name,)
        ).fetchone()[0]
        self.cursor.execute(
            'UPDATE activation_functions SET {} = ? WHERE fn_outputs = ?'.format(result_name),
            (result_value, fn_output)
        )
        self.conn.commit()


    def calculate_fisher_eigs(self, from_scratch=False, fn_names_list=None, return_eigs=False):
        """
        Calculate the eigenvalues of the Fisher information matrix for all
        activation functions in the database.  If from_scratch is True, then
        recalculate the eigenvalues for all functions.  If fn_names_list is
        provided, then only calculate the eigenvalues for those functions.
        If return_eigs is True, then immediately return the eigenvalues instead of
        storing them in the database.
        """
        if from_scratch:
            self.cursor.execute('UPDATE activation_functions SET fisher_eigs = NULL')
            self.conn.commit()

        if fn_names_list is None:
            unique_fns = self.get_unique_fn_names()
            suffix = '(all)'
        else:
            unique_fns = fn_names_list
            suffix = '(baselines)'
        pbar = tqdm(total=len(unique_fns), desc=f'Calculating Fisher eigenvalues {suffix}')

        while True:
            # Get the unique activation functions in the database
            # for which we don't have fisher eigs.
            if fn_names_list is None:
                fn_names = self.get_unique_fn_names(condition='fisher_eigs IS NULL')
                fn_names_evaluated = self.get_unique_fn_names(condition='fisher_eigs IS NOT NULL')
                pbar.update(len(fn_names_evaluated) - pbar.n)
            else:
                fn_names = fn_names_list

            # If there are no more functions to evaluate, we're done.
            if len(fn_names) == 0:
                break

            # Otherwise, choose random functions to evaluate.  We'll evaluate 100 in this job
            # to avoid calling the expensive get_unique_fn_names too many times.
            np.random.seed()
            fn_names = np.random.choice(fn_names, min(100, len(fn_names)), replace=False)
            for fn_name in fn_names:
                tf.random.set_seed(42)
                np.random.seed(42)
                try:
                    model = self.build_model(fn_name)
                    fim = FIM(model, self.samples, self.labels, self.loss)
                    eigenvalues_by_layer = fim.calculate_eigenvalues(log_scale=True)
                    if return_eigs:
                        return eigenvalues_by_layer
                    eigenvalue_cdfs = []
                    for eigenvalues, num_weights in zip(
                            eigenvalues_by_layer, self.weights_per_layer):
                        num_bins = num_weights // 100
                        bins = np.linspace(-100, 100, num_bins)
                        pdf, _, _ = plt.hist(eigenvalues, bins=bins, density=True)
                        cdf = np.cumsum(pdf)
                        cdf /= cdf[-1]
                        eigenvalue_cdfs.extend(list(cdf))
                except (ValueError, ZeroDivisionError) as err:
                    print(f'Error calculating eigenvalues for {fn_name}: {err}')
                    eigenvalue_cdfs = []

                eigenvalue_cdfs = str(eigenvalue_cdfs)

                # Add this result to the database for this function and
                # for all equivalent functions (those with the same outputs).
                self.update_for_all_equivalent_fns(fn_name, 'fisher_eigs', eigenvalue_cdfs)
                pbar.update(1)
            if fn_names_list is not None:
                break
            # Garbage collect to avoid memory leaks.
            gc.collect()
        pbar.close()
