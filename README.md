# 🎮 Rock Paper Scissors Hand Detection Game

A fun interactive game where you play rock paper scissors against the computer using hand gestures! Built with OpenCV and MediaPipe.

![Demo](demo.gif)

## 🎯 What it does

This game lets you play the classic rock paper scissors game by just showing your hand to the camera. No clicking needed - just make the gesture and the computer will detect it!

- **🪨 Rock**: Close your fist completely
- **📄 Paper**: Open your hand and show all 5 fingers  
- **✂️ Scissors**: Show just your index and middle finger

## 🚀 How to run it

1. **Clone this repo:**
   ```bash
   git clone https://github.com/MERYEMELOSMANI/rock-paper-scissors-hand-game.git
   cd rock-paper-scissors-hand-game
   ```

2. **Install the requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the game:**
   ```bash
   python rock_paper_scissors.py
   ```

4. **Have fun!** Make gestures in front of your camera and press SPACE to play!

## 🎮 Controls

- **SPACE** - Start a new round (when gesture is detected)
- **R** - Reset the game and score
- **Q** - Quit the game

## 📋 Requirements

- Python 3.7 or higher
- A webcam (built-in or external)
- Good lighting for hand detection

## 🛠️ How it works

The game uses **MediaPipe** to detect hand landmarks in real-time, then analyzes finger positions to determine which gesture you're making. It's pretty accurate once you get the hang of it!

I spent some time tweaking the finger detection logic to make it work reliably - turns out detecting scissors vs rock can be tricky depending on how you hold your fingers.

## 🐛 Troubleshooting

**Hand not detected?**
- Make sure you have good lighting
- Keep your hand clearly visible in the camera frame
- Try different hand positions

**Wrong gesture detected?**
- Make clear, distinct gestures
- For scissors, make sure only index and middle fingers are extended
- For rock, close your fist completely
- For paper, spread all fingers clearly

## 🎨 Features

- Real-time hand gesture detection
- Score tracking
- Smooth countdown animations
- Win rate statistics
- Visual feedback for detected gestures

## 📝 What I learned

This was a fun project to learn about:
- Computer vision with OpenCV
- Hand landmark detection with MediaPipe
- Game state management
- Real-time gesture recognition

The trickiest part was getting the gesture detection to be reliable - had to experiment with different finger position thresholds and add gesture stability checking.

## 🤝 Contributing

Feel free to fork this project and make it better! Some ideas:
- Add more gesture types
- Improve the UI design
- Add sound effects
- Create a tournament mode

## 📄 License

This project is open source and available under the MIT License.

---

**Made with ❤️ by Meryem El Osmani**

Hope you enjoy playing! Let me know if you have any suggestions or find any bugs! 🎮