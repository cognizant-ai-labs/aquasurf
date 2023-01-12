"""
Calculate the eigenvalues of the approximate empirical Fisher information matrix.
The Fisher is approximated as block diagonal, where each block is further decomposed
as a Kronecker product of two matrices.

See:
[1] Optimizing Neural Networks with Kronecker-factored Approximate Curvature
    (https://arxiv.org/pdf/1503.05671.pdf)
[2] A Kronecker-factored approximate Fisher matrix for convolution layers
    (https://arxiv.org/pdf/1602.01407.pdf)
"""

import logging
import numpy as np
import tensorflow as tf

from scipy.linalg import eigh


class FIM:
    def __init__(self, model, samples, labels, loss_fn):
        """
        :param model: A TF/Keras model
        :param samples: Samples from the training data
        :param labels: Ground truth labels corresponding to the samples
        :param loss_fn: The model's loss function
        """
        self.layers_with_weights = (
            tf.keras.layers.Conv2D,
            tf.keras.layers.DepthwiseConv2D,
            tf.keras.layers.Dense,
        )
        self.original_model = model
        self.samples = tf.convert_to_tensor(samples)
        self.labels = tf.convert_to_tensor(labels)
        self.loss_fn = loss_fn

        self._check_model(self.original_model)
        self.model, self.input_output_pairs = self._create_multi_output_model(self.original_model)


    def _check_model(self, model):
        """
        Ensure the Conv2D layers meet the assumptions of this class, the layers with weights
        do not have nonlinearities, and the model has no nested Models.
        """
        for layer in model.layers:
            try:
                if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    assert layer.kernel_size[0] == layer.kernel_size[1]
                    assert layer.strides[0] == layer.strides[1]
                    assert layer.data_format == 'channels_last'
                    assert layer.dilation_rate == (1, 1)
                    assert layer.groups == 1

                if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                    assert layer.depth_multiplier == 1

            except AssertionError:
                logging.exception('Layer %s has unsupported attributes', layer.name)
                raise

            if isinstance(layer, self.layers_with_weights):
                if not layer.activation is tf.keras.activations.linear:
                    logging.exception(
                        'Layer %s has activation function %s. Activation functions need to be ' \
                        'implemented in separate layers.', layer.name, layer.activation.__name__)

            if isinstance(layer, tf.keras.Model):
                logging.exception('Layer %s is a nested model, which is not supported.', layer.name)


    def _create_multi_output_model(self, model):
        """
        Expose the relevant layers of the model as outputs, and keep track of input/output pairs.
        """
        output_refs = []
        input_output_refs = []
        for layer in model.layers:
            # this layer has weights, so we need its input and output
            if isinstance(layer, self.layers_with_weights):
                layer_input_ref = layer.input.ref()
                layer_output_ref = layer.output.ref()
                input_output_refs.append((layer, layer_input_ref, layer_output_ref))
                # keep track of the references for the outputs we need
                if layer_input_ref not in output_refs:
                    output_refs.append(layer_input_ref)
                if layer_output_ref not in output_refs:
                    output_refs.append(layer_output_ref)

        # preserve the original model outputs
        for output in model.outputs:
            if output.ref() not in output_refs:
                output_refs.append(output.ref())

        # dereference to get the actual outputs
        outputs = [output_ref.deref() for output_ref in output_refs]
        model = tf.keras.Model(inputs=model.inputs, outputs=outputs)

        # convert input/output references to indices
        input_output_pairs = []
        for layer, input_ref, output_ref in input_output_refs:
            input_idx = output_refs.index(input_ref)
            output_idx = output_refs.index(output_ref)
            input_output_pairs.append((layer, input_idx, output_idx))

        return model, input_output_pairs


    def _expand_tensor(self, inpt, kernel_size, stride, padding, use_bias, is_dw_conv):
        """
        Extract the patches surrounding each spatial location and flatten them into vectors.
        The input tensor is assumed to be of shape [batch_size, height, width, channels].
        See Eq. (14) in [2].
        """
        assert len(inpt.shape) == 4

        # Zero-pad the input so the output is the same size as the input
        # We don't need to pad along the batch or channel dimensions
        if padding == 'same':
            top_pad = (kernel_size - 1) // 2
            bottom_pad = kernel_size - 1 - top_pad
            left_pad = (kernel_size - 1) // 2
            right_pad = kernel_size - 1 - left_pad
            inpt = np.pad(inpt, ((0, 0), (top_pad, bottom_pad), (left_pad, right_pad), (0, 0)))

        # At each spatial location, we need to extract and vectorize the kernel_size x kernel_size
        # patch.  Instead of iterating over spatial locations and extracting patches, we can
        # iterate over spatial offsets, shift the entire image by each offset, and then extract
        # the vectorized patch from each spatial location.  This implementation is much faster
        # than the naive one.
        shifted_inpts = []
        for i in range(kernel_size):
            for j in range(kernel_size):
                # Shift the image
                shifted_inpt = np.roll(inpt, -i, axis=1)
                shifted_inpt = np.roll(shifted_inpt, -j, axis=2)
                # We won't use these border entries
                if kernel_size != 1:
                    shifted_inpt = shifted_inpt[:, :-kernel_size+1, :-kernel_size+1, :]
                # Vectorize the shifted image
                shifted_inpt = np.reshape(shifted_inpt, (
                                          shifted_inpt.shape[0],
                                          shifted_inpt.shape[1] * shifted_inpt.shape[2],
                                          shifted_inpt.shape[3]))
                shifted_inpts.append(shifted_inpt)

        # Stacking the shifted and vectorized images produces the patches we want
        output = np.stack(shifted_inpts, axis=2)

        # If the layer has a stride greater than one, there are some spatial locations that
        # will be skipped.  We can account for the stride by creating a 2D binary mask the same
        # size as the convolved output, setting the mask to zero for the skipped locations,
        # vectorizing the mask, and then applying it to the output.
        mask_shape = (inpt.shape[1] - kernel_size + 1, inpt.shape[2] - kernel_size + 1)
        stride_mask = np.prod(np.mod(np.indices(mask_shape), stride) == 0, axis=0)
        vectorized_binary_mask = stride_mask.reshape(-1).astype(bool)
        output = output[:, vectorized_binary_mask, :]

        # Reshape the output to be a matrix
        batch_size = output.shape[0]
        spatial_locations = output.shape[1]
        spatial_offsets = output.shape[2]
        channels = output.shape[3]

        if is_dw_conv:
            # Depthwise convolutions utilize separate kernels for each channel,
            # so we can vectorize over channels in this case and just keep the spatial offsets.
            # Transpose output so that the spatial offsets are the last dimension, then reshape.
            output = np.transpose(output, (0, 1, 3, 2))
            output = np.reshape(output,
                (batch_size * spatial_locations * channels, spatial_offsets))
        else:
            output = np.reshape(output,
                (batch_size * spatial_locations, spatial_offsets * channels))

        # If the layer has a bias, prepend a homogeneous column of ones to the output.
        if use_bias:
            output = np.concatenate([np.ones((output.shape[0], 1)), output], axis=1)

        return output


    def _calculate_eigenvalues_layer(self, input_eigenvalues, output_grad_eigenvalues, log_scale):
        """
        The block of the Fisher associated with this layer is decomposed as the Kronecker
        product of the input covariance and the output gradient covariance.  See Eq. (25)
        in [2].  The eigenvalues of a Kronecker product are the pairwise products of the
        eigenvalues of the factors.  Additionally, if log-scaled eigenvalues are needed, then
        instead of computing a pairwise product and then log scaling, we can first log scale
        the eigenvalues, and then compute their pairwise sum for numerical stability.
        """
        if log_scale:
            log_input_eigenvalues = np.log(input_eigenvalues)
            log_output_grad_eigenvalues = np.log(output_grad_eigenvalues)
            eigenvalues = [e1 + e2 \
                for e1 in log_input_eigenvalues for e2 in log_output_grad_eigenvalues]
        else:
            eigenvalues = np.kron(input_eigenvalues, output_grad_eigenvalues)
        return eigenvalues


    def _calculate_eigenvalues_conv2d(self,
                                      layer,
                                      layer_input,
                                      layer_output_gradient,
                                      log_scale,
                                      is_dw_conv):
        """
        Calculate the eigenvalues associated with a Conv2D layer.  See [2].
        """
        # Expand the activations.  See Eq. (14) in [2].
        expanded_activations = self._expand_tensor(inpt=layer_input,
                                                   kernel_size=layer.kernel_size[0],
                                                   stride=layer.strides[0],
                                                   padding=layer.padding,
                                                   use_bias=layer.use_bias,
                                                   is_dw_conv=is_dw_conv)
        # Vectorize the gradient across batch and spatial locations, but keep the channel dimension
        layer_output_gradient = np.reshape(layer_output_gradient,
                                           (-1, layer_output_gradient.shape[-1]))
        batch_size = layer_input.shape[0]
        spatial_locations = int(expanded_activations.shape[0] / batch_size)

        # Compute the covariance of the input activations and output gradient
        # Note that vectorizing over the batch dimension (which we did above) and then
        # computing the product is equivalent to maintaining the batch dimension and then
        # summing the resulting matrix products together.
        # See Eq. (30-32) in [2].
        input_covariance = np.matmul(expanded_activations.T, expanded_activations) / batch_size
        output_gradient_covariance = np.matmul(layer_output_gradient.T, layer_output_gradient) / (
            batch_size * spatial_locations)

        try:
            input_eigenvalues = eigh(input_covariance, eigvals_only=True)
            output_grad_eigenvalues = eigh(output_gradient_covariance, eigvals_only=True)
            eigenvalues = self._calculate_eigenvalues_layer(input_eigenvalues,
                                                            output_grad_eigenvalues,
                                                            log_scale)
        except ValueError:
            logging.warning('Could not calculate eigenvalues for layer %s. Using zeros instead.',
                            layer.name)
            eigenvalues = np.zeros(input_covariance.shape[0] * output_gradient_covariance.shape[0])

        return eigenvalues


    def _calculate_eigenvalues_dense(self, layer, layer_input, layer_output_gradient, log_scale):
        """
        Calculate the eigenvalues associated with a Dense layer.  See [1].
        """
        # Transpose the input and output gradient so that each column represents a single sample.
        layer_input = layer_input.numpy().T
        layer_output_gradient = layer_output_gradient.numpy().T
        # If the layer has a bias, prepend a homogeneous row of ones to the input.
        if layer.use_bias:
            layer_input = np.concatenate([np.ones((1, layer_input.shape[1])), layer_input], axis=0)
        batch_size = layer_input.shape[0]

        # See Eq. (1) in [1].
        input_covariance = np.matmul(layer_input, layer_input.T) / batch_size
        output_gradient_covariance = np.matmul(
            layer_output_gradient, layer_output_gradient.T) / batch_size

        try:
            input_eigenvalues = eigh(input_covariance, eigvals_only=True)
            output_grad_eigenvalues = eigh(output_gradient_covariance, eigvals_only=True)
            eigenvalues = self._calculate_eigenvalues_layer(input_eigenvalues,
                                                            output_grad_eigenvalues,
                                                            log_scale)
        except ValueError:
            logging.warning('Could not calculate eigenvalues for layer %s. Using zeros instead.',
                            layer.name)
            eigenvalues = np.zeros(input_covariance.shape[0] * output_gradient_covariance.shape[0])

        return eigenvalues


    def calculate_eigenvalues(self, log_scale=False):
        """
        Calculate the eigenvalues of the Fisher information matrix.
        """
        # Pass the data through the model.
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.samples)
            outputs = self.model(self.samples)
            # The predictions are assumed to be the final output of the model.
            predictions = outputs[-1]
            loss = self.loss_fn(self.labels, predictions)

        # The eigenvalues of a block diagonal matrix are simply the eigenvalues of each block.
        # Each block corresponds to a layer with weights, so we simply iterate over these layers
        # and calculate the eigenvalues associated with each one.
        eigenvalues = []
        for layer, layer_input_idx, layer_output_idx in self.input_output_pairs:
            layer_input = outputs[layer_input_idx]
            layer_output = outputs[layer_output_idx]
            layer_output_gradient = tape.gradient(loss, layer_output)

            if layer_output_gradient is None:
                logging.warning('Could not calculate eigenvalues for layer %s. ' \
                    'Skipping this layer.', layer.name)
                continue

            if isinstance(layer, tf.keras.layers.Conv2D):
                eigenvalues.append(self._calculate_eigenvalues_conv2d(layer,
                                                                      layer_input,
                                                                      layer_output_gradient,
                                                                      log_scale=log_scale,
                                                                      is_dw_conv=False))

            elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                eigenvalues.append(self._calculate_eigenvalues_conv2d(layer,
                                                                      layer_input,
                                                                      layer_output_gradient,
                                                                      log_scale=log_scale,
                                                                      is_dw_conv=True))

            else:
                assert isinstance(layer, tf.keras.layers.Dense)
                eigenvalues.append(self._calculate_eigenvalues_dense(layer,
                                                                     layer_input,
                                                                     layer_output_gradient,
                                                                     log_scale=log_scale))

        return eigenvalues
