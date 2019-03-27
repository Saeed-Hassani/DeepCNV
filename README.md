# DeepCNV
A novel cancer type classifier based on deep learning and copy number variants
Demo
معماری برنامه:

فرمت ورودی به برنامه:

|Gene_1 | Gene_2 |   ...  | Gene_n| Class |
|=========================================|
|   +2  |   -1   |   ...  |   -2  |   C1  |
|   -1  |   +2   |   ...  |   -1  |   C2  |
|    .  |    .   |    .   |    .  |   .   |
|    0  |   -2   |   ...  |    0  |   C1  |
|   +1  |    0   |   ...  |   +2  |   C3  |

Input format of the program
در این برنامه ستونها به عنوان ژنها در نظر گرفته میشوند و سطرهای مقادیر سطح CNV میباشد. همچنین مقدار ستون آخر نشان دهنده کلاس سرطان میباشد.
In this program, the column consider as gene and the rows is as the level of CNV values and the last column is as type of the cancer
پیش نیاز ها
Pre-Requirements
برای نصب برنامه باید کتابخانه های زیر نصب شود.
For installing the app, the following package must be installed.
چگونگی نصب برنامه
How to install the program
پارامترها:
Parameters:
LSTM Units: 
Dropout: 
Test Size: 

Validation Size: 
Epoch:

Batch Size: 
Desired Feature:
Shuffle: 
This section of the program, show the architecture of lstm in the program
نمونه خروجی:
Sample of the output
