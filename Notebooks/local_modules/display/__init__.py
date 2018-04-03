
from google.colab import files
import IPython
from IPython.core.display import HTML

def configure_plotly_browser_state():
  display(IPython.core.display.HTML('''
        <script>
          alert('Hi there!');
        </script>
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))
 

def upload_stuff():
  uploaded = files.upload()

  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
    
    with open(fn, 'w') as f:
      f.write(uploaded[fn])
  
  return uploaded
#upload_stuff()

def wav_player(filepath):
    """ will display html 5 player for compatible browser

    Parameters :
    -------4/AACUBaoJnIh-l2J9iHfQB8L4osBcgU8dHf1SnCupweWgwyGAlaEWoyU-----
    fsilence_filterilepath : relative filepath with respect to the notebook directory ( where the .ipynb are not cwd)
               of the file to play

    The browser need to know how to play wav through html5.

    there is no autoplay to prevent file playing when the browser opens
    """
    
    src = """
    <head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8">
    <title>Simple Test</title>
    </head>
    
    <body>
    <audio controls="controls" style="width:600px" >
      <source src="files/%s" type="audio/wav" />
      Your browser does not support the audio element.
    </audio>
    </body4/AACUBaoJnIh-l2J9iHfQB8L4osBcgU8dHf1SnCupweWgwyGAlaEWoyU>
    """%(filepath)
    display(HTML(src))