# pytorch-ergonomics
Improving the pytorch experience on [Algorithmia](https://algorithmia.com]) and beyond.

*Caveat - The ergonomic changes in this package have been checked for compatibilty with python 3.5.4, or the `python3` language on Algorithmia.
Some components might not work for other versions.*
## improving the Algorithmia pytorch runtime experience
Pytorch works on Algorithmia out of the box, however the default pytorch wheel files
 are more of a "kitchen sink" than a efficient minimalist package. This is not an issue when experimenting locally on your own machine, but
 because of how Algorithmia loads algorithm dependencies these large packages can _really_ slow down algorithm execution.

 Thankfully we've come up with a solution! Here is a step by step guide on how to go about improving your algorithm runtime experience:

 * First - add one of our modifed versions of pytorch as a dependency to your algorithm:
   * CPU mode only -
     * https://s3.amazonaws.com/algorithmia-wheels/torch_cpu-0.3.1b0+2b47480-cp35-cp35m-linux_x86_64.whl
     * https://s3.amazonaws.com/algorithmia-wheels/torch_cpu-0.4.0a0+a3d08de-cp35-cp35m-linux_x86_64.whl
   * GPU mode only -
     * https://s3.amazonaws.com/algorithmia-wheels/torch_gpu-0.3.1b0+2b47480-cp35-cp35m-linux_x86_64.whl
 * Second - We currently require a workaround to be used when interacting with pytorch using our trimmed down wheels, otherwise you'll get a whole bunch of nasty import errors.
   * import the `algorithm_improvement` script into your project, and execute the `execute_workaround` function.

If you're in need of the `torchvision` module, you'll probably want to use our compiled version as opposed to what's available on pypi, you can find it here:

https://s3.amazonaws.com/algorithmia-wheels/torchvision-0.2.0-py2.py3-none-any.whl

 For a concrete example of how to apply this to an algorithm, look at our [pytorchDemo](https://algorithmia.com/algorithms/algorithmiahq/pytorchDemo).

## Improving the model processing experience
The pytorch `torch.load()` and `torch.save()` tools are great at what they do, serializing big complex DAGs with potentially
 millions of super critical arrays can be pretty tricky. A design decision was made where the default torch model loader protocol does _not_ save the
 network definitions required to execute your code along with the model itself.

In most instances, using these two tools with your static source code is more than reasonable.
Most of the time, your model definitions are static and you're purely looking
to explore ways to improve your model accuracy and perform analysis on your training process.
In production however, this changes dramatically - backwards compatability matters *alot*.

The `model_ergonmics` script is designed to address this, and allows you to save your network definition module
along with your torch model. This means that your updated algorithm can still work with torch models that might
not have been backwards compatible historically! You can even import your torch model into an algorithm and run a
forward pass without having to copy your source code by hand, making the cross-algorithm model loading experience DRY (Don't Repeat Yourself!)

Concrete example for this is coming, for a simple example please check out our `tests` file.
