# Communication Patterns code

This folder contains the code used to analyze data in the [CDR](https://en.wikipedia.org/wiki/Call_detail_record) format.  

1. [Friends and Family](http://realitycommons.media.mit.edu/friendsdataset4.html) data set. This set of call comes from the USA.
2. A UK data set using students transitioning from high school to university. This data set is not present in this folder, mostly due to privacy.
3. An Italian data set using phone calls from young families. This data is also not present in this folder, for the same reasons as the UK case.
    
    
All notebooks with the code used for data analysis are named `YYYYMM.ipynb`; where `YYYY` is the year and `MM` the month of the analysis. The reason to separate the code is to better organize and look for a particular piece of information. Also, given the amount of ideas that lead nowhere, it saves time when loading and running a particular notebook.

## General functions

All notebooks use the `Python` module `phonecalls`, which contains all the functions with some comments explaining the arguments and expected outcomes. Make sure to load it whenever using the code contained here.
