# Windows Installation with Miniconda

* I've installed Miniconda 3.8
* From find I've looked for "Miniconda" and started the application "Anaconda Prompt (Miniconda)"

Now, we need to create a virtual env for the project. I did it with Pycharm based on the installed Miniconda. 
This venv is created in 
```
C:\daten\projects\GNN_experiments\venv\Scripts
``` 
so I entered this folder. The virtual env was created under the main GNN_experiments folder which means 
that we need to go to this folder.

Now, I've activated this venv with 
```
activate
```
and I got the following
```
(venv) (base) C:\daten\projects\GNN_experiments\venv\Scripts>
```

Now, many python modules are missing. So I've installed this packages (behind the Bosch firewall) 
with the following command:
```
pip.exe --proxy https://USER:PASSWORD@10.4.103.143:8080 install networkx
```

## Install torch geometric on Lenovo X380 windows machine with torch==1.7.1+cpu
```
pip.exe --proxy https://USER:PASS@10.4.103.143:8080 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip.exe --proxy https://USER:PASS@10.4.103.143:8080 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip.exe --proxy https://USER:PASS@10.4.103.143:8080 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
pip.exe --proxy https://USER:PASS@10.4.103.143:8080 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
# This is must since the pytorch geometric by itself needs to download things and need proxy.
set http_proxy=http://<USER>:<PASSWORD>@10.4.103.143:8080
set https_proxy=http://<USER>:<PASSWORD>@10.4.103.143:8080
# 
pip.exe --proxy https://USER:PASS@10.4.103.143:8080 install torch-geometric -f https://pytorch-geometric.com/whl/torch-1.7.0+cpu.html
```

Note: consider to reinstall ```openssl``` from here: https://slproweb.com/products/Win32OpenSSL.html 
