fileId=1loIh6AYygDALfID1_Ou2z6AwQB1MEjOY
fileName=fasterrcnn_12211511_0.701052458187_torchvision_pretrain.pth
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${fileId}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${fileId}" -o ${fileName}
