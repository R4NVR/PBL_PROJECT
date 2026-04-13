import speech_recognition as sr
import pyautogui
import os
import time

recognizer = sr.Recognizer()

print("Press ENTER to speak a song name")

while True:

    input("Press ENTER to activate voice control...")

    print("Listening...")

    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:

        song = recognizer.recognize_google(audio)

        print("Song detected:", song)

        # open spotify
        os.startfile("spotify")

        # wait for spotify to load
        time.sleep(5)

        # open search
        pyautogui.hotkey("ctrl","l")

        time.sleep(1)

        # type song name
        pyautogui.write(song)

        time.sleep(1)

        # press enter
        pyautogui.press("enter")

        print("Playing:", song)

    except Exception as e:

        print("Voice not recognized")


