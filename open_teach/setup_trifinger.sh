#!/bin/bash

sudo apt install hostapd
cp hostapd.conf /etc/hostapd/
if [ -f /etc/default/hostapd ]; then
    # Use sed to modify the line
    sudo sed -i 's|^#DAEMON_CONF=.*|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd
else
    echo "/etc/default/hostapd file does not exist."
    exit 1
fi

sudo apt install isc-dhcp-server
cp dhcpd.conf /etc/dhcp/dhcpd.conf
FILE="/etc/default/isc-dhcp-server"
INTERFACE="wlx347de44f8dcb"

# Check for the existence of the file
if [ -f "$FILE" ]; then
    # Use sed to find and replace INTERFACESv4 and INTERFACESv6 lines or add them if not found
    if grep -q "^INTERFACESv4=" "$FILE"; then
        sudo sed -i.bak "s|^INTERFACESv4=.*|INTERFACESv4=\"$INTERFACE\"|" "$FILE"
    else
        echo "INTERFACESv4=\"$INTERFACE\"" | sudo tee -a "$FILE"
    fi

    if grep -q "^INTERFACESv6=" "$FILE"; then
        sudo sed -i.bak "s|^INTERFACESv6=.*|INTERFACESv6=\"$INTERFACE\"|" "$FILE"
    else
        echo "INTERFACESv6=\"$INTERFACE\"" | sudo tee -a "$FILE"
    fi

    echo "Modified $FILE to set INTERFACESv4 and INTERFACESv6 to \"$INTERFACE\""
else
    echo "$FILE does not exist. Exiting."
    exit 1
fi

FILE2="/etc/network/interfaces"

ADDRESS="10.10.0.1"
NETMASK="255.255.255.0"

# Check for the existence of the file
if [[ -f "$FILE2" ]]; then
    # Check if the configuration already exists
    if grep -q "auto $INTERFACE" "$FILE2"; then
        echo "Configuration for $INTERFACE already exists in $FILE2"
    else
        # Append the configuration to the file
        echo -e "\nauto $INTERFACE\niface $INTERFACE inet static\n    address $ADDRESS\n    netmask $NETMASK" | sudo tee -a "$FILE2"
        echo "Added configuration for $INTERFACE to $FILE2"
    fi
else
    echo "$FILE2 does not exist. Exiting."
    exit 1
fi