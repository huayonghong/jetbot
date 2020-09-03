export JETBOT_VERSION=r32.4.3

JUPYTER_WORKSPACE=${1:-$HOME}  # default to $HOME

sudo docker run -it -d \
	--restart always \
	--runtime nvidia \
       	--network host \
	--privileged \
	--device /dev/video* \
	--volume /dev/bus/usb:/dev/bus/usb \
	--volume /tmp/argus_socket:/tmp/argus_socket \
	-p 8888:8888 \
	-v ${JUPYTER_WORKSPACE}:/workspace \
	--workdir /workspace \
	--name=jetbot_uni \
	--memory=1000m \
	--memory-swap=3G \
	nvcr.io/ea-linux4tegra/jetbot_uni:${JETBOT_VERSION}
