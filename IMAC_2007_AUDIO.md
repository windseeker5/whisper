# Audio Setup for 2007 iMac

## Built-in Microphone

**Answer**: `Built-in Audio Analog Stereo` - This is your built-in microphone!

When you run the configuration, select this option:
```
5. Built-in Audio Analog Stereo [pyaudio]
    Recommended: 48000 Hz, 1 channels
```

## Other Audio Devices Explained

Your 2007 iMac shows these audio devices:

### Built-in Microphone
- **Built-in Audio Analog Stereo** ← **THIS IS IT!**
  - The microphone built into your iMac's display
  - Sample Rate: 48000 Hz
  - **Best choice for voice transcription**

### Physical Audio Ports (Back of iMac)
These are the physical jacks on the back:

1. **HDA Intel: ALC889A Analog (hw:0,0)**
   - Line-in/Mic-in port (back of iMac)
   - Only active if you plug in an external microphone

2. **HDA Intel: ALC889A Digital (hw:0,1)**
   - Digital audio input (S/PDIF)
   - Not useful for microphone

3. **HDA Intel: ALC889A Alt Analog (hw:0,2)**
   - Alternative analog input
   - Not commonly used

### Software/Virtual
4. **pipewire**
   - Software audio routing
   - Can be used but built-in is better

## Recommended Configuration

```bash
python bin/voice_transcriber.py --config
```

When asked:
1. **Microphone**: Select `5` (Built-in Audio Analog Stereo)
2. **Backend**: Already configured to Vosk ✓
3. **Language**: Select `1` (Auto-detect) or `2` (English)

## Testing Your Microphone

After configuration:

```bash
# Test audio levels
python bin/voice_transcriber.py --test-audio

# You should see audio levels respond when you speak
```

## Troubleshooting

### No audio detected
1. Check system volume/mute: `pavucontrol` or `alsamixer`
2. Verify microphone is not muted in system settings
3. Try speaking louder near the display (mic is in the top bezel)

### Audio levels too low
- The built-in iMac mic is not very sensitive
- Position yourself 1-2 feet from the display
- Speak clearly and at normal volume
- The app has auto-gain control to boost quiet audio

## Your iMac's Microphone Location

```
┌─────────────────────────────────────┐
│           ● ← Microphone            │  Top bezel of display
│                                     │
│         [  Display  ]               │
│                                     │
│                                     │
└─────────────────────────────────────┘
```

The microphone is in the top center of your iMac's display bezel (the small hole near the camera if you have one).

## Performance Notes

The built-in microphone on 2007 iMacs is adequate but not great quality:
- ✓ Good enough for Vosk transcription
- ✓ Auto-gain control will help boost levels
- ⚠ Background noise can be picked up
- 💡 Tip: Speak 1-2 feet from display for best results
