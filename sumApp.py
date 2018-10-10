from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
 
#import textrank as tr
 
class Summary(GridLayout):
 
     def __init__(self, **kwargs):
        super(Summary, self).__init__(**kwargs)
        self.cols = 2
        self.add_widget(Label(text='Input Text:'))
        self.input = TextInput(multiline=False)
        self.add_widget(self.input)
        
        
        self.add_widget(Label(text='Generated Summary:'))
        self.output = TextInput(multiline=False)
        self.add_widget(self.output)
        
        
class MyApp(App):
 
    def build(self):
        return Summary()
 
 
if __name__ == '__main__':
    MyApp().run()