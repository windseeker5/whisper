#!/bin/bash
"""
Wayland/Hyprland Environment Setup Script
Sets up environment variables required for voice transcriber service
to work properly with Wayland and Hyprland compositor.
"""

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | systemd-cat -t voice-transcriber-env
}

# Export Wayland environment variables
export XDG_SESSION_TYPE=wayland
export QT_QPA_PLATFORM=wayland
export GDK_BACKEND=wayland
export SDL_VIDEODRIVER=wayland
export CLUTTER_BACKEND=wayland

# Hyprland-specific variables
if command -v hyprctl >/dev/null 2>&1; then
    export HYPRLAND_INSTANCE_SIGNATURE=$(hyprctl instances -j | jq -r '.[0].instance')
    log_message "Hyprland instance detected: $HYPRLAND_INSTANCE_SIGNATURE"
fi

# Get display and session information
if [ -n "$WAYLAND_DISPLAY" ]; then
    export WAYLAND_DISPLAY="$WAYLAND_DISPLAY"
    log_message "Using Wayland display: $WAYLAND_DISPLAY"
else
    # Try to detect Wayland display
    for display in wayland-0 wayland-1; do
        if [ -S "/run/user/$(id -u)/$display" ]; then
            export WAYLAND_DISPLAY="$display"
            log_message "Auto-detected Wayland display: $display"
            break
        fi
    done
fi

# Set XDG runtime directory if not set
if [ -z "$XDG_RUNTIME_DIR" ]; then
    export XDG_RUNTIME_DIR="/run/user/$(id -u)"
    log_message "Set XDG_RUNTIME_DIR: $XDG_RUNTIME_DIR"
fi

# Audio system detection and setup
if command -v pipewire >/dev/null 2>&1; then
    export PULSE_RUNTIME_PATH="$XDG_RUNTIME_DIR/pulse"
    log_message "PipeWire audio system detected"
elif command -v pulseaudio >/dev/null 2>&1; then
    export PULSE_RUNTIME_PATH="$XDG_RUNTIME_DIR/pulse"
    log_message "PulseAudio system detected"
fi

# DBus session setup
if [ -z "$DBUS_SESSION_BUS_ADDRESS" ]; then
    export DBUS_SESSION_BUS_ADDRESS="unix:path=$XDG_RUNTIME_DIR/bus"
    log_message "Set DBUS_SESSION_BUS_ADDRESS: $DBUS_SESSION_BUS_ADDRESS"
fi

log_message "Wayland/Hyprland environment setup completed"

# Verify critical components
log_message "Environment verification:"
log_message "  WAYLAND_DISPLAY: ${WAYLAND_DISPLAY:-NOT_SET}"
log_message "  XDG_RUNTIME_DIR: ${XDG_RUNTIME_DIR:-NOT_SET}"
log_message "  DBUS_SESSION_BUS_ADDRESS: ${DBUS_SESSION_BUS_ADDRESS:-NOT_SET}"
log_message "  HYPRLAND_INSTANCE_SIGNATURE: ${HYPRLAND_INSTANCE_SIGNATURE:-NOT_SET}"