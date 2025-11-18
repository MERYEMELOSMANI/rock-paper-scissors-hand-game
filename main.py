import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math

# Rock Paper Scissors Game with Hand Detection
# A fun computer vision project using MediaPipe
# Challenge the computer with your hand gestures!

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1,
    min_detection_confidence=0.8, 
    min_tracking_confidence=0.8
)

# Game colors for better visual experience
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
    """Main game class that handles all the game logic"""
    
    def __init__(self):
        self.player_score = 0
        self.computer_score = 0
        self.rounds_played = 0
        self.game_state = "waiting"  # waiting, countdown, playing, result
        self.countdown_start = 0
        self.result_start = 0
        self.player_choice = ""
        self.computer_choice = ""
        self.result = ""
        self.choices = ["ROCK", "PAPER", "SCISSORS"]
        self.choice_emojis = {"ROCK": "🪨", "PAPER": "📄", "SCISSORS": "✂️"}
        
    def get_computer_choice(self):
        """Generate random computer choice"""
        return random.choice(self.choices)
    
    def determine_winner(self, player, computer):
        """Classic Rock Paper Scissors logic"""
        if player == computer:
            return "TIE"
        elif (player == "ROCK" and computer == "SCISSORS") or \
             (player == "PAPER" and computer == "ROCK") or \
             (player == "SCISSORS" and computer == "PAPER"):
            return "PLAYER WINS"
        else:
            return "COMPUTER WINS"
    
    def start_countdown(self):
        """Initialize countdown phase"""
        self.game_state = "countdown"
        self.countdown_start = time.time()
    
    def update_countdown(self):
        """Update countdown timer"""
        elapsed = time.time() - self.countdown_start
        if elapsed >= 3.0:  # 3 second countdown
            self.game_state = "playing"
            return True
        return False
    
    def play_round(self, player_choice):
        """Execute a complete game round"""
        self.player_choice = player_choice
        self.computer_choice = self.get_computer_choice()
        self.result = self.determine_winner(player_choice, self.computer_choice)
        
        # Update score tracking
        if self.result == "PLAYER WINS":
            self.player_score += 1
        elif self.result == "COMPUTER WINS":
            self.computer_score += 1
        
        self.rounds_played += 1
        self.game_state = "result"
        self.result_start = time.time()
    
    def show_result_complete(self):
        """Check if result display time is over"""
        return time.time() - self.result_start >= 3.0
    
    def reset_round(self):
        """Reset for next round"""
        self.game_state = "waiting"
        self.player_choice = ""
        self.computer_choice = ""
        self.result = ""

def detect_hand_gesture(landmarks):
    """Analyze hand landmarks to detect Rock, Paper, Scissors gestures"""
    if not landmarks:
        return "NONE"
    
    # Get important finger positions
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
    
    # Check which fingers are extended
    fingers_up = []
    
    # Thumb - special case
    fingers_up.append(thumb_tip[1] < thumb_ip[1])
    
    # Other fingers - check if tip is higher than pip joint
    finger_tips = [index_tip, middle_tip, ring_tip, pinky_tip]
    finger_pips = [index_pip, middle_pip, ring_pip, pinky_pip]
    
    for tip, pip in zip(finger_tips, finger_pips):
        fingers_up.append(tip[1] < pip[1])
    
    # Count extended fingers
    extended_count = sum(fingers_up)
    
    # Gesture classification based on finger count and position
    if extended_count == 0:
        return "ROCK"
    elif extended_count == 5:
        return "PAPER"
    elif extended_count == 2 and fingers_up[1] and fingers_up[2]:  # Index and middle
        return "SCISSORS"
    elif extended_count == 2 and fingers_up[1] and fingers_up[3]:  # Index and ring
        return "SCISSORS"
    else:
        return "UNKNOWN"

def draw_gesture_guide(img):
    """Display visual instructions for gestures"""
    guide_y = img.shape[0] - 200
    
    # Create guide background
    cv2.rectangle(img, (20, guide_y - 30), (img.shape[1] - 20, img.shape[0] - 20), 
                 (50, 50, 50), -1)
    cv2.rectangle(img, (20, guide_y - 30), (img.shape[1] - 20, img.shape[0] - 20), 
                 WHITE, 2)
    
    # Add title
    cv2.putText(img, "GESTURE GUIDE", (30, guide_y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
    
    # Show gesture instructions
    gestures = [
        "🪨 ROCK = CLOSED FIST",
        "📄 PAPER = OPEN HAND (5 FINGERS)",
        "✂️ SCISSORS = INDEX + MIDDLE EXTENDED"
    ]
    
    for i, gesture in enumerate(gestures):
        cv2.putText(img, gesture, (30, guide_y + 25 + i * 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)

def draw_scoreboard(img, game):
    """Display current game score and status"""
    # Score background
    cv2.rectangle(img, (20, 20), (400, 120), (40, 40, 40), -1)
    cv2.rectangle(img, (20, 20), (400, 120), WHITE, 3)
    
    # Game title
    cv2.putText(img, "ROCK PAPER SCISSORS", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2)
    
    # Current scores
    cv2.putText(img, f"PLAYER: {game.player_score}", (30, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    cv2.putText(img, f"COMPUTER: {game.computer_score}", (200, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    cv2.putText(img, f"ROUNDS: {game.rounds_played}", (30, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

def draw_countdown(img, game):
    """Show animated countdown before round starts"""
    h, w = img.shape[:2]
    elapsed = time.time() - game.countdown_start
    countdown_num = 3 - int(elapsed)
    
    if countdown_num > 0:
        # Large countdown number
        cv2.putText(img, str(countdown_num), (w//2 - 50, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 5, ORANGE, 10)
        
        # Round indicator
        cv2.putText(img, f"ROUND {game.rounds_played + 1}", (w//2 - 100, h//2 + 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, WHITE, 3)
    else:
        cv2.putText(img, "SHOW YOUR HAND!", (w//2 - 200, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, RED, 5)

def draw_result(img, game):
    """Display round results with choices and winner"""
    h, w = img.shape[:2]
    center_x = w // 2
    center_y = h // 2
    
    # Result background
    cv2.rectangle(img, (center_x - 250, center_y - 150), 
                 (center_x + 250, center_y + 150), (30, 30, 30), -1)
    cv2.rectangle(img, (center_x - 250, center_y - 150), 
                 (center_x + 250, center_y + 150), WHITE, 3)
    
    # Player choice with emoji
    player_emoji = game.choice_emojis.get(game.player_choice, "❓")
    cv2.putText(img, f"YOU: {player_emoji}", (center_x - 240, center_y - 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, GREEN, 3)
    cv2.putText(img, game.player_choice, (center_x - 240, center_y - 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    
    # Computer choice with emoji
    computer_emoji = game.choice_emojis.get(game.computer_choice, "❓")
    cv2.putText(img, f"COMPUTER: {computer_emoji}", (center_x - 240, center_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, RED, 3)
    cv2.putText(img, game.computer_choice, (center_x - 240, center_y + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    
    # Winner announcement
    result_color = GREEN if game.result == "PLAYER WINS" else RED if game.result == "COMPUTER WINS" else YELLOW
    cv2.putText(img, game.result, (center_x - 100, center_y + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, result_color, 4)

def draw_waiting_screen(img, current_gesture):
    """Show waiting screen with current gesture detection"""
    h, w = img.shape[:2]
    center_x = w // 2
    center_y = h // 2
    
    # Main instruction
    cv2.putText(img, "MAKE A GESTURE TO START!", (center_x - 250, center_y - 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)
    
    # Current gesture feedback
    if current_gesture != "NONE" and current_gesture != "UNKNOWN":
        gesture_emoji = {"ROCK": "🪨", "PAPER": "📄", "SCISSORS": "✂️"}.get(current_gesture, "❓")
        cv2.putText(img, f"DETECTED: {gesture_emoji}", (center_x - 150, center_y - 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, CYAN, 2)
        cv2.putText(img, current_gesture, (center_x - 80, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
        
        # Ready indicator
        cv2.putText(img, "PRESS SPACE TO PLAY!", (center_x - 180, center_y + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 2)
    else:
        cv2.putText(img, "SHOW YOUR HAND", (center_x - 150, center_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, ORANGE, 3)

def draw_game_stats(img, game):
    """Show additional game statistics"""
    h, w = img.shape[:2]
    stats_x = w - 300
    
    # Win percentage calculation
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

    print("🎮 ROCK PAPER SCISSORS GAME ACTIVATED! ✂️🪨📄")
    print("=== GAME CONTROLS ===")
    print("🪨 ROCK = Close your fist")
    print("📄 PAPER = Open your hand (all 5 fingers)")
    print("✂️ SCISSORS = Show index and middle finger")
    print("SPACE = Start game when gesture is detected")
    print("R = Reset game")
    print("Q = Quit game")
    print("Get ready to challenge the computer! 🎯")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # Create dark overlay for better visibility
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
                
                # Draw hand skeleton
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
                                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))
                
                # Detect current gesture
                detected_gesture = detect_hand_gesture(landmarks)
                
                # Gesture stability check to avoid flickering
                if detected_gesture == last_gesture:
                    gesture_stable_time += 1
                else:
                    gesture_stable_time = 0
                    last_gesture = detected_gesture
                
                # Update current gesture if stable enough
                if gesture_stable_time > 10:  # ~10 frames for stability
                    current_gesture = detected_gesture
        else:
            current_gesture = "NONE"
            gesture_stable_time = 0
        
        # Game state machine
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
            # Brief transition state
            pass
        
        elif game.game_state == "result":
            draw_result(frame, game)
            if game.show_result_complete():
                game.reset_round()
        
        # Draw UI elements
        draw_scoreboard(frame, game)
        draw_gesture_guide(frame)
        draw_game_stats(frame, game)
        
        # Current gesture indicator
        if current_gesture != "NONE" and current_gesture != "UNKNOWN":
            gesture_emoji = game.choice_emojis.get(current_gesture, "❓")
            cv2.putText(frame, f"CURRENT: {gesture_emoji}", (w - 250, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, CYAN, 2)
            cv2.putText(frame, current_gesture, (w - 200, 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        
        cv2.imshow('Rock Paper Scissors Game ✂️🪨📄', frame)
        
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
            print("🔄 Game reset!")

    cap.release()
    cv2.destroyAllWindows()
    print(f"🎮 Game Over! Final Score - Player: {game.player_score}, Computer: {game.computer_score}")
    print("Thanks for playing Rock Paper Scissors! ✂️🪨📄✨")

if __name__ == "__main__":
    main()