# Cloudchamber Wallpaper
A live android wallpaper of a cloud chamber.

I decided to use the [Processing Language](https://android.processing.org) for creating the wallpaper. Since I was not very familiar with Processing, I decided to create a proof of concept in python and port it over.

This was supposed to be a Slime Wallpaper inspired by [a video by Sebastian Lague](#credit), but I could not get it to run smoothly on a phone, so I decided to pivot to making a live background of a cloud chamber simulation, since the slime reminded me of this.

## Overview

This wallpaper simulates the particle tracks seen in a cloud chamber physics experiment. The simulation features:

- Alpha and beta particle simulations
- Particle diffusion and evaporation effects
- Touch interaction to create new particles

## Installation Instructions

### Prerequisites
- [Processing](https://processing.org/download/) 
- [Android Mode for Processing](https://android.processing.org/install.html)

### Building the Wallpaper

1. Open the `CloudChamberWallpaper.pde` file in Processing
2. Switch to Android Mode in Processing (using the dropdown in the top-right corner)
3. Connect your Android device via USB with USB debugging enabled
4. Click the "Run on Device" button in Processing

### Manual Installation

After building, you can find the APK file in:
```
CloudChamberWallpaper/build/android/bin/CloudChamberWallpaper.apk
```

Transfer this file to your Android device and install it.

After installation:
1. Go to your Android home screen
2. Long press on an empty area
3. Select "Wallpapers"
4. Select "Live Wallpapers"
5. Find and select "Cloud Chamber"
6. Tap "Set Wallpaper"


## Customization

You can modify the following parameters to change the appearance:
- Particle color, size, and velocity in the `createAgent()` method
- Diffusion and evaporation rates
- Particle creation frequency

## Credit

This Project was heavily inspired by [this video](https://www.youtube.com/watch?v=X-iSQQgOd1A) by [Sebastian Lague](https://www.youtube.com/@SebastianLague).

The Video: https://www.youtube.com/watch?v=X-iSQQgOd1A
The corresponding GitHub Repo: https://github.com/SebLague/Slime-Simulation/tree/main
His Youtube Channel: https://www.youtube.com/@SebastianLague
His GitHub Account: https://github.com/SebLague