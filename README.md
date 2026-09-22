## Make SD card with Raspberry Pi OS

Use Raspberry Pi Imager to flash OS on a microSD.

Go with Raspberry Pi OS (64-bit) (the standard Desktop one, not Lite, not Full)

Hostname (e.g. something distinct from your other Pis, like thermal-p1v2)\
Username and password use the same:\
  ale\
  cameratrapale\
Enable SSH\
Do not enable Raspberry Pi Connect





If you want to avoid typing the sudo password every time:

```
sudo visudo
```

add the following to set the timeout to 120 minutes:

    Defaults timestamp_timeout=120

Setting this to -1 will make the password last until you reboot or close the terminal.\
Save and exit the editor.

To disable password completely:\
Add the following line (replace your_username with your actual username):

    your_username ALL=(ALL) NOPASSWD: ALL





# File transfer between Raspberry Pi and Windows PC


To transfer files from Raspberry Pi to a Windows PC, we can connect them via an Ethernet cable and create a little network between them. Then we can use WinSCP to transfer files between devices.

Here below are the main instructions, summed up from https://chatgpt.com/share/6a99302e-2fac-83ed-a487-1ad906cb88ee


## Assign IP to Raspberry Pi

On the Raspberry Pi, please run these one at a time:

    sudo ip link set eth0 up

then:

    sudo ip addr add 192.168.50.2/24 dev eth0

Then:

    ip addr show eth0

You should see something like:

```
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> ...
    inet 192.168.50.2/24 scope global eth0
    inet6 fe80::a00a:bcb4:b8c2:84...
```

This assign that IP to Raspberry Pi, but it is temporary, until next reboot!\
First let's make sure you can connect via WinSCP, then we'll make this IP permanent.



## Give your Windows Ethernet adapter an IP in the same subnet

On Windows:

Open Settings → Network & Internet → Advanced network settings\
Click More network adapter options\
Right-click your Ethernet adapter → Properties\
Select Internet Protocol Version 4 (TCP/IPv4)\
Click Properties\
Select Use the following IP address\

Enter:

IP address:      192.168.50.1\
Subnet mask:     255.255.255.0\
Default gateway: [leave blank]\
DNS server:      [leave blank]\

Click OK.

This creates a tiny private network:

Windows laptop                  Raspberry Pi
192.168.50.1  ─── Ethernet ───  192.168.50.2

No router or internet is necessary.





Test it on Windows

On Windows PowerShell:

    ping 192.168.50.2

You should get replies.\
Then:

    Test-NetConnection 192.168.50.2 -Port 22

You want:

    TcpTestSucceeded : True



Connect with WinSCP

Use:

File protocol: SFTP\
Host name:     192.168.50.2\
Port:          22\
Username:      <your Raspberry Pi username>\
Password:      <your Pi password>\

Then click Login.


If you have issues, try to sudo reboot the Raspi.


### Make that IP permanent on Raspberry Pi

Run these commands on the Raspberry Pi.\
Here is for "Wired connection 1". Further below is for "netplan-eth0".

1. Set the permanent static IP

   sudo nmcli connection modify "Wired connection 1" ipv4.method manual ipv4.addresses 192.168.50.2/24

Since this Ethernet connection is just for directly connecting to your Windows laptop, explicitly disable the default gateway and DNS on this connection:

    sudo nmcli connection modify "Wired connection 1" ipv4.never-default yes
    sudo nmcli connection modify "Wired connection 1" ipv6.method disabled

Then activate the updated connection:

    sudo nmcli connection down "Wired connection 1"
    sudo nmcli connection up "Wired connection 1"



In case of "netplan-eth0":

```
sudo nmcli connection modify "netplan-eth0" ipv4.method manual ipv4.addresses 192.168.50.2/24
```
```
sudo nmcli connection modify "netplan-eth0" ipv4.never-default yes
sudo nmcli connection modify "netplan-eth0" ipv6.method disabled
```
```
sudo nmcli connection down "netplan-eth0"
sudo nmcli connection up "netplan-eth0"
```  

2. Verify

Run:

    ip addr show eth0

You should see:

    inet 192.168.50.2/24

You can also check with:

    nmcli device show eth0
    
3. Test from Windows

Make sure your Windows Ethernet adapter is still configured as:

IP address:  192.168.50.1\
Subnet mask: 255.255.255.0

Then in PowerShell:

    ping 192.168.50.2

and:

    Test-NetConnection 192.168.50.2 -Port 22

If everything is working, TcpTestSucceeded should be True.

4. The real test: reboot the Pi

Once the above works, reboot:

    sudo reboot

After it restarts, reconnect the Ethernet cable and check:

    ip addr show eth0

If you still see:

    inet 192.168.50.2/24

then the IP is now permanent. Your WinSCP connection can always use:

Host: 192.168.50.2\
Port: 22\
Protocol: SFTP\
Username: ale

This configuration should also leave your seewiesen Wi-Fi connection alone, so the Pi can use Wi-Fi normally while eth0 remains your dedicated direct connection to the laptop.





# Install code

```
sudo apt update
sudo apt install python3-full python3-venv python3-opencv python3-numpy libusb-1.0-0-dev
```

Now create an environment and activate it. If needed, use cd to move to the directory where you want the environment to be created.
```
cd home/ale
```
```
python3 -m venv venv
source venv/bin/activate
```
```
pip install numpy pyusb opencv-python matplotlib
```

### Install repository
```
git clone https://github.com/lachioma/ThermalMaster-p1-camera
cd ThermalMaster-p1-camera
pip install -e .
```

USB permissions for the camera (needed for the camera to work):
```
sudo tee /etc/udev/rules.d/99-p3-ir.rules << EOF
# P1 camera
SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45c2", MODE="0666"
# P3 camera
SUBSYSTEM=="usb", ATTR{idVendor}=="3474", ATTR{idProduct}=="45a2", MODE="0666"
EOF
sudo udevadm control --reload-rules
sudo udevadm trigger
```



## Test code and usage

Activate venv environment:

```
source venv/bin/activate
```

Run:

```
python p3_viewer.py --model p1
```

Commands:

https://github.com/jvdillon/p3-ir-camera



## Connect Raspberry Pi Camera Module 2 before powering on Raspi

<img width="1400" height="800" alt="image" src="https://github.com/user-attachments/assets/e1f9d48d-9360-4203-aee2-455b6df65cfe" />


You'll need internet connection. 

To use Neuroguest, you first need to connect to another network to update time and clock, otherwise you will not be able to be authenticated to Neuroguest.
The easiest way is to use your phone mobile hotspot. Connect just to update the clock. Then you can connect to Neuroguest after requesting a ticket. 
Neuroguest porta: https://guestportal.neuro.mpg.de/login.html

Then, for the camera, run:

    sudo apt update && sudo apt full-upgrade
    
Then you need to reboot:

    sudo reboot
    
Verify the camera: 

    rpicam-hello --list-cameras


Install picamera2

python3 -m venv --system-site-packages ~/venv
source /venv/bin/activate
python3 -c "from picamera2 import Picamera2; print('ok')"


