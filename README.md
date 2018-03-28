# pytorch-algorithmia
Improving the pytorch experience on Algorithmia.

*Caveat - all of these changes were made specifically for python 3.5.4, or the `python3` language on Algorithmia, some components might not work for other versions.*
## improving the loading experience
Pytorch works on Algorithmia out of the box, however the default pytorch wheel files
 are more of a "kitchen sink" than a efficient minimalist package. This is not an issue when experimenting locally on your own machine, but
 because of how Algorithmia loads algorithm dependencies these large packages can _really_ slow down algorithm execution.

 Thankfully we've come up with a solution! Here is a step by step guide on how to go about improving your algorithm runtime experience:

 * First - add one of our modifed versions of pytorch as a dependency to your algorithm:
   * CPU mode only -
 https://s3.amazonaws.com/algorithmia-wheels/torch_cpu-0.3.1b0+2b47480-cp35-cp35m-linux_x86_64.whl
 https://s3.amazonaws.com/algorithmia-wheels/torch_cpu-0.4.0a0+a3d08de-cp35-cp35m-linux_x86_64.whl
   * GPU mode only -
 https://s3.amazonaws.com/algorithmia-wheels/torch_gpu-0.3.1b0+2b47480-cp35-cp35m-linux_x86_64.whl
 * Second - We currently require a workaround to be used when interacting with pytorch using our trimmed down wheels, otherwise you'll get a whole bunch of nasty import errors.
   * import the `algorithm_improvement` script into your project, and execute the `execute_workaround` function.

 For a concrete example of how to apply this to an algorithm, look at our [pytorchDemo][https://algorithmia.com/algorithms/algorithmiahq/pytorchDemo].


 Let us know what you think! As we find new ways of improving the pytorch experience this package will evolve.