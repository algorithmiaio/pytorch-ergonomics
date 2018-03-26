# pytorch-algorithmia
Improving the pytorch experience on Algorithmia.


## improving the loading experience
Pytorch works on Algorithmia out of the box, however the default pytorch wheel files
 are more of a "kitchen sink" than a efficient minimalist package. This is not an issue when experimenting locally on your own machine, but
 because of how Algorithmia loads algorithm dependencies these large packages can _really_ slow down algorithm execution.

 Thankfully we've come up with a solution!

 First - add this version of pytorch as a dependency to your algorithm:
 `https://s3.amazonaws.com/algorithmia-wheels/torch-0.3.0b0+af3964a-cp35-cp35m-linux_x86_64.whl`

 Second - We currently require a workaround to be used when interacting with pytorch, otherwise you'll get a whole bunch of nasty import errors.
 import the `load_improvement` script into your project, and execute the `execute_workaround` function.

 That's it! For more information, take a look at our `tests.py` file for a full end-to-end example.