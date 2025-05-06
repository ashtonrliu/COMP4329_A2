#READ ME

## guidelines:
- variable names should be in camel case (variableName)
- function names should be in pascal case (FunctionName)
- global variables should be in all caps snake case (GLOBAL_VARIABLE)
- global variables should be treated as read-only
- hard coded values should be global variables
- variable names should not be shortened (use dataframe instead of df. Some exceptions apply ie i and j for counters)
- variable names and function names should be descriptive at a glance
- avoid function side effects (a method that is meant to print out information should not alter outside variables)
- independent code blocks should be seperated into individual functions (if you think you need a comment to explain a code block, you should extract it into a seperate function)
