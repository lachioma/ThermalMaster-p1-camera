#!/bin/bash
# Prepare the Raspberry Pi for an unattended field deployment:
# - starts (and enables for future boots) the thermal recorder
# - turns off Wi-Fi, Bluetooth, and wired Ethernet
# - switches the boot target to console (no desktop GUI)
#
# HDMI is deliberately left alone - you need it to plug in a display when
# interrupting field mode to run normal_mode.sh. (A second-HDMI power-saving
# attempt via max_framebuffers=1 was tried and reverted - confirmed via
# vcgencmd/modetest that it reached the firmware but had no effect on the
# active CRTC count under this OS's KMS driver, so it wasn't doing anything.)
#
# Reverse everything with normal_mode.sh.
set -e

echo "Blocking Wi-Fi and Bluetooth radios..."
sudo rfkill block wifi
sudo rfkill block bluetooth

echo "Disabling the Bluetooth daemon..."
sudo systemctl disable --now bluetooth.service

# daemon-reload once up front picks up any edits to either .service file
# below since it was last installed; restart (rather than enable --now)
# for both guarantees the current settings take effect even if either was
# already running.
sudo systemctl daemon-reload

echo "Telling NetworkManager to leave eth0 alone..."
# Without this, NetworkManager re-manages (and re-activates) eth0 on every
# boot regardless of what field-mode-net.service just set it to - this is
# what was actually causing eth0 to come back up after a reboot.
NM_CONF="/etc/NetworkManager/conf.d/99-unmanage-eth0.conf"
if [ -d /etc/NetworkManager/conf.d ]; then
    printf '[keyfile]\nunmanaged-devices=interface-name:eth0\n' | sudo tee "$NM_CONF" > /dev/null
    sudo systemctl reload NetworkManager 2>/dev/null || true
else
    echo "  /etc/NetworkManager/conf.d not found (not using NetworkManager?); skipping."
fi

echo "Disabling wired Ethernet..."
sudo systemctl enable field-mode-net.service
sudo systemctl restart field-mode-net.service

echo "Switching the boot target to console (no desktop GUI)..."
# No-op with a harmless warning if this is already a Lite (no-desktop) install.
sudo systemctl set-default multi-user.target

echo "(Re)starting the thermal recorder..."
sudo systemctl enable thermal-recorder.service
sudo systemctl restart thermal-recorder.service

echo
echo "Recorder status:"
sudo systemctl status thermal-recorder.service --no-pager -l | head -n 10

echo
echo "Field mode is set. The recorder is already running now."
echo "Reboot so the console boot target takes effect for future power-ups:"
echo "  sudo reboot"
