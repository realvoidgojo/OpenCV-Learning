from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
from ctypes import cast, POINTER

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

print("Audio device found")
print(f"- Muted: {bool(volume.GetMute())}")
print(f"- Volume level: {volume.GetMasterVolumeLevel()} dB")
print(f"- Volume range: {volume.GetVolumeRange()[0]} dB - {volume.GetVolumeRange()[1]} dB")
volume.SetMasterVolumeLevel(-20.0, None)