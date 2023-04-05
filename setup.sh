pip install braceexpand
pip install timm
pip install google-cloud-storage
pip install tensorboardX
pip install ftfy
pip install regex
pip install setuptools==59.5.0
pip install transformers==4.27.2
pip install diffusers==0.14.0
pip install git+https://github.com/openai/CLIP.git
pip install torchmetrics
pip install open_clip_torch
# pip install google-cloud-storage
# pip install -r requirements.txt

if test -f "/root/.bashrc"; then
  echo """PATH="/export/home/google-cloud-sdk/bin:\$PATH"""" >> /root/.bashrc
  source /root/.bashrc
fi
