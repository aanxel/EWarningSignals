# Guides for the creation of a generic library
https://medium.com/analytics-vidhya/how-to-create-a-python-library-7d5aea80cc3f

# Library Tests execution
I haven't tested these commands, because Pycharm already gives a tool to execute all tests inside a folder.
```cmd
python -m unittest -v tests/multiplication_tests.py
python -m unittest -v tests
```

# Build of the Library
This command is old and in process of being deprecated.
(**https://towardsdatascience.com/deep-dive-create-and-publish-your-first-python-library-f7f618719e14**)
```cmd
python setup.py sdist bdist_wheel
```

Instead, the proper command to build and export your library is the following one 
(**https://packaging.python.org/en/latest/tutorials/packaging-projects/**).
```cmd
python -m build
```
This command will create two new folders named **dist** and **projectFolder.egg-info**. The first one will contain the
real generated code from your own library as a **.tar.gz** and **.whl**. From this one we can extract the **.whl** file
and try to install it in your own environment.

As a recommendation, each time you want to build the library after any changes, I strongly suggest to delete both
folders previously mentioned and all its content (**dist** and **projectFolder.egg-info**). This is because it might
have cached some files and the output won't be the expected one.

# Install library in your environment
Once you have the **.whl** file you can install it in your own environment. In case you are using **pipenv** you should
launch the following command:
```cmd
pipenv install file_name.whl
```

In case you are having troubles installing it, I suggest the following things. First, make sure to create a new
environment with no dependencies of any type or uninstall all your previous packages.
```cmd
pipenv uninstall --all
```
Make sure that there are no packages or lines under the paragraph [packages]. So the Pipfile should look like this:

```Pipfile
[[source]]
url = "https://pypi.python.org/simple"
verify_ssl = true
name = "pypi"

[dev-packages]

[scripts]

[requires]
python_version="3.9"

[packages]
```

In case there are any lines after [packages] you should uninstall it manually for example:
```cmd
pipenv uninstall ewarningsignals
```

Other helpful commands might be:
(**https://stackoverflow.com/questions/40183108/python-packages-hash-not-matching-whilst-installing-using-pip**)
```cmd
pipenv clean
pipenv --clear
pipenv lock
pipenv install
```

Other helpful option might be completely remove the pipenv environment with the firsts command and install the binary
again.
```cmd
pipenv --rm
```

# Documentation and Docstring of the library
@FUTURO

# Extra
Place to download library binaries in **whl** format.
(https://www.lfd.uci.edu/~gohlke/pythonlibs/)


Check if gdal is installed.
```
from osgeo import gdal
```
