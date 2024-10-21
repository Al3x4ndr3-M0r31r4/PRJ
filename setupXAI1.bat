:: Run this script on the anaconda prompt
:: The first path is the location of the .yml file that contains the information of which libraries to install
:: The second path is the path where you want to put the anaconda environment and its name 
:: Directories with spaces are a cause of problems for the environment, make sure the environment
:: does not include spaces anywhere its path.
call conda env create --file "D:\Program Files\ISEL\MEIM\tese\envXAI.yml" --prefix D:\ProgramFiles\envXAISetup
:: This command activates the environment so that modifications can be made
call conda activate D:\ProgramFiles\envXAISetup
pause