from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Rectangle, Color
from kivy.uix.filechooser import FileChooserIconView
import cv2
import os
import numpy as np

class WelcomeScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Image above the welcome
        self.logo_image = Image(source=r"C:\Users\Admin\OneDrive\Documents\Dermatter app\dermatter.jpg", size_hint=(1, None), height=460)
        layout.add_widget(self.logo_image)
        
        # Welcome message
        welcome_label = Label(
            text="Welcome to Dermatter",
            font_size='32sp',
            color=(0, 0.5, 1, 1),  # Pastel blue color
            bold=True,
            size_hint_y=None,
            height=50,
            halign='center'
        )
        layout.add_widget(welcome_label)
        
        # Start button
        start_button = Button(
            text="Let's Start",
            size_hint_y=None,
            height=50,
            background_color=(0.2, 0.6, 1, 1),  # Pastel blue
            color=(1, 1, 1, 1),  # White text
            font_size='18sp',
            bold=True,
        )
        start_button.bind(on_press=self.start_comparison)
        layout.add_widget(start_button)
        
        self.add_widget(layout)
        
        with self.canvas.before:
            Color(1, 1, 1, 1)  # White background color
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self._update_rect, pos=self._update_rect)
    
    def _update_rect(self, instance, value):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def start_comparison(self, instance):
        self.manager.current = 'main'

class MainScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layout = BoxLayout(orientation='vertical', padding=20, spacing=20)
        
        # Text input for the user image path
        self.image_path_input = TextInput(
            hint_text='Select the path of the user image',
            size_hint_y=None,
            height=40,
            multiline=False,
            padding=[10, 10, 10, 10],
            background_color=(0.9, 0.9, 0.9, 1),  # Light gray
            foreground_color=(0, 0, 0, 1)
        )
        self.layout.add_widget(self.image_path_input)
        
        # Button to open file chooser
        self.choose_file_button = Button(
            text='Choose Image',
            size_hint_y=None,
            height=50,
            background_color=(0.2, 0.6, 1, 1),  # Pastel blue
            color=(1, 1, 1, 1),  # White text
            font_size='18sp',
            bold=True,
        )
        self.choose_file_button.bind(on_press=self.open_file_chooser)
        self.layout.add_widget(self.choose_file_button)
        
        # Button to start comparison
        self.compare_button = Button(
            text='Compare Images',
            size_hint_y=None,
            height=50,
            background_color=(0.2, 0.6, 1, 1),  # Pastel blue
            color=(1, 1, 1, 1),  # White text
            font_size='18sp',
            bold=True,
        )
        self.compare_button.bind(on_press=self.compare_images)
        self.layout.add_widget(self.compare_button)
        
        # Label to show the result
        self.result_label = Label(
            text='',
            size_hint_y=None,
            height=40,
            font_size='18sp',
            halign='center',
            color=(0, 0, 0, 1)  # Black text
        )
        self.layout.add_widget(self.result_label)
        
        # Image widget to display the result
        self.result_image = Image(size_hint_y=None, height=400)
        self.layout.add_widget(self.result_image)
        
        self.add_widget(self.layout)
        
        with self.canvas.before:
            Color(1, 1, 1, 1)  # White background color
            self.rect = Rectangle(size=self.size, pos=self.pos)

        self.bind(size=self._update_rect, pos=self._update_rect)
    
    def _update_rect(self, instance, value):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def open_file_chooser(self, instance):
        content = BoxLayout(orientation='vertical')
        file_chooser = FileChooserIconView()
        file_chooser.path = r"C:\Users\Admin\OneDrive\Documents\Dermatter app\ref"
        content.add_widget(file_chooser)

        def on_select_file(instance, selection):
            if selection:
                self.image_path_input.text = selection[0]  # Update the text input with the selected file path
                popup.dismiss()

        button = Button(text='Select', size_hint_y=None, height=40)
        button.bind(on_press=lambda x: on_select_file(button, file_chooser.selection))
        content.add_widget(button)

        popup = Popup(title='Select Image', content=content, size_hint=(0.9, 0.9))
        popup.open()

    def compare_images(self, instance):
        user_img_path = self.image_path_input.text
        folder_path = r"C:\Users\Admin\OneDrive\Documents\Dermatter app\skin"

        if not user_img_path:
            self.show_popup('Error', 'Please select the path of the user image.')
            return

        # Load user image
        user_img = cv2.imread(user_img_path)
        if user_img is None:
            self.show_popup('Error', f"Image at {user_img_path} not found or could not be read.")
            return
    
    # Convert user image to grayscale
        gray_user_img = cv2.cvtColor(user_img, cv2.COLOR_BGR2GRAY)

    # Create ORB detector
        orb = cv2.ORB_create()
        kp_user, des_user = orb.detectAndCompute(gray_user_img, None)

    # Initialize variables to track the best match
        best_match_img = None
        best_match_percentage = 0
        best_match_image_name = ""

    # Iterate through all images in the folder
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                continue

        # Convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = orb.detectAndCompute(gray_img, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_user, des)
            num_matches = len(matches)
            total_keypoints_user = len(kp_user)
            total_keypoints_img = len(kp)

            if total_keypoints_user == 0 or total_keypoints_img == 0:
                match_percentage = 0
            else:
                match_percentage = (num_matches / max(total_keypoints_user, total_keypoints_img)) * 100

            if match_percentage > best_match_percentage:
                best_match_percentage = match_percentage
                best_match_img = img
                best_match_image_name, _ = os.path.splitext(filename)  # Extract filename without extension

        if best_match_img is not None:
            best_img_gray = cv2.cvtColor(best_match_img, cv2.COLOR_BGR2GRAY)
            kp_best, des_best = orb.detectAndCompute(best_img_gray, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches_best = bf.match(des_user, des_best)
            matches_best = sorted(matches_best, key=lambda x: x.distance)
            img_matches = cv2.drawMatches(user_img, kp_user, best_match_img, kp_best, matches_best[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Convert the image to a format Kivy can use
            img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
            img_matches = np.rot90(img_matches)  # Rotate to fit Kivy's coordinate system
            img_matches = np.flip(img_matches, axis=0)  # Flip to correct orientation
            img_matches_texture = self._array_to_texture(img_matches)
        
        # Update the result
            self.result_label.text = f"Best match image: {best_match_image_name}\nBest match percentage: {best_match_percentage:.2f}%"
            self.result_image.texture = img_matches_texture
        else:
            self.result_label.text = "No matches found."

    def _array_to_texture(self, array):
        h, w, c = array.shape
        texture = Texture.create(size=(w, h), colorfmt='rgb')
        texture.blit_buffer(array.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        return texture

    def show_popup(self, title, message):
        popup = Popup(
            title=title,
            content=Label(text=message, halign='center'),
            size_hint=(0.8, 0.4),
            background_color=(1, 1, 1, 1),  # White background
            title_color=(0.2, 0.2, 0.2, 1)  # Dark gray title
        )
        popup.open()

class ImageComparatorApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(WelcomeScreen(name='welcome'))
        sm.add_widget(MainScreen(name='main'))
        return sm

if __name__ == '__main__':
    ImageComparatorApp().run()
