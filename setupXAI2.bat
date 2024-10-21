:: Since the version of instalation of shap, for some reason, is not correctly instaled,
:: this alterations must be made.
:: First activate the environment created by the first script
call conda activate D:\ProgramFiles\envXAISetup
:: Then uninstall the shap version that was instaled via pip
call pip uninstall -y shap
:: After pip uninstall the enviroment is activated again
call conda activate D:\ProgramFiles\envXAISetup
:: The correct instalation of shap, using conda is made
call conda install -y -c conda-forge shap==0.42.1
pause
