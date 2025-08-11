# FastTD3
reproduce the project FastTD3 on my laptop

#environment
my device --unbuntu20.04

#approach

1.install nvidia driver 
https://www.google.com/url?sa=E&q=https.nvidia.com%2FDownload%2Findex.aspx

2.install CUDA 12
https://www.google.com/url?sa=E&q=https%3A%2F%2Fdeveloper.nvidia.com%2Fcuda-12-5-0-download-archive

3.Configure environment variables:
After installation, you need to add the CUDA path to your environment variables. Edit your ~/.bashrc or ~/.zshrc file (depending on the shell you are using):
  
  nano ~/.bashrc

Add the following lines at the end of the file (please adjust according to your actual installation path and version):

  export PATH=/usr/local/cuda-12.5/bin${PATH:+:${PATH}}
  export LD_LIBRARY_PATH=/usr/local/cuda-12.5/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

After saving the file, run the following command to apply the changes:
    
  source ~/.bashrc

