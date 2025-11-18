"""
Rock Paper Scissors Hand Detection Game
Created by: Meryem El Osmani
A fun interactive game where you play rock paper scissors against the computer using hand gestures!
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math

# Initialize MediaPipe for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, 
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Define colors for UI
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)
CYAN = (255, 255, 0)

class RockPaperScissorsGame:
    """Main game class that handles game logic and state"""
    
    def __init__(self):
        self.player_score = 0
        self.computer_score = 0
        self.rounds_played = 0
        self.game_state = "waiting"  # States: waiting, countdown, playing, result
        self.countdown_start = 0
        self.result_start = 0
        self.player_choice = ""
        self.computer_choice = ""
        self.result = ""
        self.choices = ["ROCK", "PAPER", "SCISSORS"]
        self.choice_emojis = {"ROCK": "🪨", "PAPER": "📄", "SCISSORS": "✂️"}
        
    def get_computer_choice(self):
        """Get a random choice for the computer"""
        return random.choice(self.choices)
    
    def determine_winner(self, player, computer):
        """Figure out who won the round based on classic RPS rules"""
        if player == computer:
            return "TIE"
        elif (player == "ROCK" and computer == "SCISSORS") or \
             (player == "PAPER" and computer == "ROCK") or \
             (player == "SCISSORS" and computer == "PAPER"):
            return "PLAYER WINS"
        else:
            return "COMPUTER WINS"
    
    def start_countdown(self):
        """Begin the countdown before showing hands"""
        self.game_state = "countdown"
        self.countdown_start = time.time()
    
    def update_countdown(self):
        """Check if countdown is finished"""
        elapsed = time.time() - self.countdown_start
        if elapsed >= 3.0:  # 3 second countdown
            self.game_state = "playing"
            return True
        return False
    
    def play_round(self, player_choice):
        """Execute one round of the game"""
        self.player_choice = player_choice
        self.computer_choice = self.get_computer_choice()
        self.result = self.determine_winner(player_choice, self.computer_choice)
        
        # Update scores based on who won
        if self.result == "PLAYER WINS":
            self.player_score += 1
        elif self.result == "COMPUTER WINS":
            self.computer_score += 1
        
        self.rounds_played += 1
        self.game_state = "result"
        self.result_start = time.time()
    
    def show_result_complete(self):
        """Check if we've shown the result long enough"""
        return time.time() - self.result_start >= 3.0
    
    def reset_round(self):
        """Reset everything for the next round"""
        self.game_state = "waiting"
        self.player_choice = ""
        self.computer_choice = ""
        self.result = ""

def detect_hand_gesture(landmarks):
    """
    Detect rock, paper, or scissors gesture from hand landmarks
    This was tricky to get right - had to experiment with different finger positions!
    """
    if not landmarks:
        return "NONE"
    
    # Get the important finger positions
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    index_tip = landmarks[8]
    index_pip = landmarks[6]
    middle_tip = landmarks[12]
    middle_pip = landmarks[10]
    ring_tip = landmarks[16]
    ring_pip = landmarks[14]
    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]
    
    # Check which fingers are extended (tip higher than pip joint)
    fingers_up = []
    
    # Thumb is a bit different
    fingers_up.append(thumb_tip[1] < thumb_ip[1])
    
    # Other fingers - check if tip is above the pip joint
    finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
    finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
    
    for tip, pip in zip(finger_tips, finger_pips):
        fingers_up.append(tip[1] < pip[1])
    
    # Count how many fingers are extended
    extended_count = sum(fingers_up)
    
    # Classify the gesture based on extended fingers
    if extended_count == 0:
        return "ROCK"
    elif extended_count == 5:
        return "PAPER"
    elif extended_count == 2 and fingers_up[1] and fingers_up[2]:  # Index and middle
        return "SCISSORS"
    elif extended_count == 2 and fingers_up[1] and fingers_up[3]:  # Index and ring (alternative)
        return "SCISSORS"
    else:
        return "UNKNOWN"

def draw_gesture_guide(img):
    """Draw instructions for the gestures at the bottom of screen"""
    guide_y = img.shape[0] - 200
    
    # Background box for instructions
    cv2.rectangle(img, (20, guide_y - 30), (img.shape[1] - 20, img.shape[0] - 20), 
                 (50, 50, 50), -1)
    cv2.rectangle(img, (20, guide_y - 30), (img.shape[1] - 20, img.shape[0] - 20), 
                 WHITE, 2)
    
    # Title
    cv2.putText(img, "HOW TO PLAY", (30, guide_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    
    # Instructions for each gesture
    gestures = [
        "🪨 ROCK = Close your fist completely",
        "📄 PAPER = Open your hand (show all 5 fingers)", 
        "✂️ SCISSORS = Show index and middle finger only"
    ]
    
    for i, gesture in enumerate(gestures):
        cv2.putText(img, gesture, (30, guide_y + 25 + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

def draw_scoreboard(img, game):
    """Draw the current score and game info"""
    # Score background
    cv2.rectangle(img, (20, 20), (400, 120), (40, 40, 40), -1)
    cv2.rectangle(img, (20, 20), (400, 120), WHITE, 3)
    
    # Game title
    cv2.putText(img, "ROCK PAPER SCISSORS", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
    
    # Current scores
    cv2.putText(img, f"YOU: {game.player_score}", (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    cv2.putText(img, f"COMPUTER: {game.computer_score}", (200, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    cv2.putText(img, f"ROUNDS: {game.rounds_played}", (30, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

def draw_countdown(img, game):
    """Draw the countdown animation before each round"""
    h, w = img.shape[:2]
    elapsed = time.time() - game.countdown_start
    countdown_num = 3 - int(elapsed)
    
    if countdown_num > 0:
        # Big countdown number in center
        cv2.putText(img, str(countdown_num), (w//2 - 50, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, ORANGE, 10)
        
        # Animated countdown circle
        center = (w//2, h//2 + 100)
        radius = int(80 * (1 - (elapsed % 1)))
        cv2.circle(img, center, radius, YELLOW, 5)
    else:
        # Time to show your hand!
        cv2.putText(img, "SHOW YOUR HAND!", (w//2 - 200, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 5)

def draw_result(img, game):
    """Display the results of the round"""
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    
    # Result display background
    cv2.rectangle(img, (center_x - 250, center_y - 150), 
                 (center_x + 250, center_y + 150), (30, 30, 30), -1)
    cv2.rectangle(img, (center_x - 250, center_y - 150), 
                 (center_x + 250, center_y + 150), WHITE, 3)
    
    # Show what each player chose
    player_emoji = game.choice_emojis.get(game.player_choice, "❓")
    cv2.putText(img, f"YOU: {player_emoji}", (center_x - 240, center_y - 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
    cv2.putText(img, game.player_choice, (center_x - 240, center_y - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    
    computer_emoji = game.choice_emojis.get(game.computer_choice, "❓")
    cv2.putText(img, f"COMPUTER: {computer_emoji}", (center_x - 240, center_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
    cv2.putText(img, game.computer_choice, (center_x - 240, center_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    
    # Show who won this round
    result_color = GREEN if game.result == "PLAYER WINS" else RED if game.result == "COMPUTER WINS" else YELLOW
    cv2.putText(img, game.result, (center_x - 100, center_y + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 4)

def draw_waiting_screen(img, current_gesture):
    """Show waiting screen with gesture detection"""
    center_x = img.shape[1] // 2
    center_y = img.shape[0] // 2
    
    # Instructions
    cv2.putText(img, "MAKE A GESTURE TO START!", (center_x - 250, center_y - 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)
    
    # Show what gesture is currently detected
    if current_gesture != "NONE" and current_gesture != "UNKNOWN":
        gesture_emoji = {"ROCK": "🪨", "PAPER": "📄", "SCISSORS": "✂️"}.get(current_gesture, "❓")
        cv2.putText(img, f"DETECTED: {gesture_emoji}", (center_x - 150, center_y - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, CYAN, 2)
        cv2.putText(img, current_gesture, (center_x - 80, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
        
        # Ready to play message
        cv2.putText(img, "PRESS SPACE TO PLAY!", (center_x - 180, center_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)
    else:
        cv2.putText(img, "SHOW YOUR HAND", (center_x - 150, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, ORANGE, 3)

def draw_game_stats(img, game):
    """Show additional game statistics"""
    stats_x = img.shape[1] - 300
    
    # Calculate win percentage
    if game.rounds_played > 0:
        win_rate = (game.player_score / game.rounds_played) * 100
        cv2.putText(img, f"WIN RATE: {win_rate:.1f}%", (stats_x, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
    
    # Current game status
    status_color = GREEN if game.game_state == "playing" else ORANGE
    cv2.putText(img, f"STATUS: {game.game_state.upper()}", (stats_x, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

def main():
    """Main game loop"""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    game = RockPaperScissorsGame()
    current_gesture = "NONE"
    gesture_stable_time = 0
    last_gesture = "NONE"

    print("🎮 ROCK PAPER SCISSORS GAME STARTING!")
    print("=== CONTROLS ===")
    print("🪨 ROCK = Close your fist")
    print("📄 PAPER = Open your hand (all 5 fingers)")
    print("✂️ SCISSORS = Show index and middle finger")
    print("SPACE = Start game when gesture is detected")
    print("R = Reset game")
    print("Q = Quit game")
    print("Let's see if you can beat the computer! 🎯")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame from camera. Check your camera connection!")
            break
        
        frame = cv2.flip(frame, 1)  # Mirror the image
        h, w, _ = frame.shape
        
        # Create darker overlay for better text visibility
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        current_time = time.time()
        
        # Hand detection and gesture recognition
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [(int(l.x * w), int(l.y * h)) for l in hand_landmarks.landmark]
                
                # Draw hand skeleton with nice colors
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
                
                # Detect what gesture is being made
                detected_gesture = detect_hand_gesture(landmarks)
                
                # Make sure gesture is stable before accepting it
                if detected_gesture == last_gesture:
                    gesture_stable_time += 1
                else:
                    gesture_stable_time = 0
                    last_gesture = detected_gesture
                
                # Update current gesture if it's been stable for enough frames
                if gesture_stable_time > 10:  # About 10 frames for stability
                    current_gesture = detected_gesture
        else:
            current_gesture = "NONE"
            gesture_stable_time = 0
        
        # Handle different game states
        if game.game_state == "waiting":
            draw_waiting_screen(frame, current_gesture)
        
        elif game.game_state == "countdown":
            if game.update_countdown():
                # Countdown finished, capture player gesture
                if current_gesture in game.choices:
                    game.play_round(current_gesture)
                else:
                    game.play_round("ROCK")  # Default to rock if no valid gesture
            else:
                draw_countdown(frame, game)
        
        elif game.game_state == "playing":
            # This state is very brief, just for transition
            pass
        
        elif game.game_state == "result":
            draw_result(frame, game)
            if game.show_result_complete():
                game.reset_round()
        
        # Draw all the UI elements
        draw_scoreboard(frame, game)
        draw_gesture_guide(frame)
        draw_game_stats(frame, game)
        
        # Show current gesture in top right corner
        if current_gesture != "NONE" and current_gesture != "UNKNOWN":
            gesture_emoji = game.choice_emojis.get(current_gesture, "❓")
            cv2.putText(frame, f"CURRENT: {gesture_emoji}", (w - 250, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
            cv2.putText(frame, current_gesture, (w - 200, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        
        cv2.imshow('Rock Paper Scissors Hand Game', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord(' '):  # Space to start game
            if game.game_state == "waiting" and current_gesture in game.choices:
                game.start_countdown()
        elif key == ord('r'):  # Reset game
            game = RockPaperScissorsGame()
            current_gesture = "NONE"
            print("🔄 Game reset! Starting fresh...")

    cap.release()
    cv2.destroyAllWindows()
    print(f"🎮 Game Over! Final Score - You: {game.player_score}, Computer: {game.computer_score}")
    if game.player_score > game.computer_score:
        print("🏆 Congratulations! You won overall! 🎉")
    elif game.computer_score > game.player_score:
        print("💻 The computer won this time. Better luck next time!")
    else:
        print("🤝 It's a tie! Great game!")
    print("Thanks for playing Rock Paper Scissors! ✂️🪨📄✨")

if __name__ == "__main__":
    main()