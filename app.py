from dash import Dash, dcc, html, Input, Output, callback,ctx,State
import plotly.graph_objects as go
import plotly.express as px
import os
import gym
from Agents import *
from collections import deque
from dash_extensions.enrich import Trigger, FileSystemCache
from tqdm import tqdm
import time


def moving_average(x, span=100):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
fsc = FileSystemCache("cache_dir")
fsc1 = FileSystemCache("cache_dir")
fsc2 = FileSystemCache("cache_dir")
fsc3 = FileSystemCache("cache_dir")
fsc4 = FileSystemCache("cache_dir")
fsc5 = FileSystemCache("cache_dir")


fsc1.set('progressbar','0')
zeros_img=np.full((400,600,3),255,dtype=np.uint8)
fsc.set('images',zeros_img)
fsc2.set('stop','0')
fsc3.set('status','idle')
fsc4.set('plotreward',[])
app.layout = html.Div([
    html.Div(dcc.Dropdown(id='models-option',
        options=[
            {'label':'Lunar Lander','value':'LunarLander'},
            {'label':'Cart Pole','value':'CartPole'}],
            value='LunarLander'
    )),
    html.Div(id='env-div',children=[dcc.Graph(id='env-plot')]),
    html.Div(id='reward-div',children=[dcc.Graph(id='reward-plot')]),
    html.Div(html.Progress(id='progressbar',value='0',max='100')),
    html.Div(id='progressbar-msgs'),
    html.Div(html.Button('Train',id='train-btn',n_clicks=0)),
    html.Div(html.Button('Reset',id='reset-btn',n_clicks=0)),
    html.Div(id='status-msg'),
    html.Div(id='msgs'),
    html.Div(id='stop-msg'),
    dcc.Interval(id='refresher',interval=100,n_intervals=0),
    dcc.Interval(id='barloader',interval=100,n_intervals=0),
    dcc.Interval(id='statusloader',interval=100,n_intervals=0)
])


@callback(Output('msgs', 'children'),            
                [Input('train-btn','n_clicks'),
                Input('reset-btn','n_clicks'),
                State('models-option','value')])
def display_value(train_clicks,reset_clicks,model_option):
    
    if ctx.triggered_id=='train-btn':
        fsc2.set('stop','0')
        if model_option=='LunarLander':
            episodes=2000
            batch_size=64
            memory_size=100
            env=gym.make('LunarLander-v2',render_mode='rgb_array')
            agent=DeepQAgent3(env,max_memory_size=memory_size,learning_rate=1e-3,discount_factor=0.999,epsilon_greedy=1)
            state=env.reset()[0]
        elif model_option=='CartPole':
            episodes=200
            batch_size=32
            memory_size=100
            env=gym.make('CartPole-v1',render_mode='rgb_array')
            agent=DeepQAgent(env)
            state=env.reset()[0]
       

        print("Filling memory")
        fsc3.set('status','Filling replay memory')
        for i in range(memory_size):
            action=agent.choose_action(state)
            next_state,reward,done,_,_=env.step(action)
            agent.remember((state,action,reward,next_state,done))
            img=env.render()
            perc=100*(i+1)/memory_size
            #print(perc)
            fsc1.set('progressbar',str(perc))

            if done:
                state=env.reset()[0]
            else:
                state=next_state
            time.sleep(0.1)
        total_rewards,losses=[],[]

        fsc3.set('status','Training')
        print('Training')
        fsc1.set('progressbar','0')


        for e in range(episodes):
            if fsc2.get('stop')=='1':
                break

            state=env.reset()[0]
            rewards=0
            done=False
            #for s in range(500):
            steps=0
            while not done :
                action=agent.choose_action(state)
                next_state,reward,done,_,_=env.step(action)
                agent.remember((state,action,reward,next_state,done))
                img=env.render()
                fsc.set("images", img)
                state=next_state
                rewards+=reward
                loss=agent.replay(batch_size)
                losses.append(loss)
                steps+=1
                if done:
                    total_rewards.append(rewards)
                    print(f'Episode: {e} Total reward:{rewards} Epsilon:{agent.epsilon} steps: {steps}')
                    break
            fsc1.set('progressbar',str(100*(e+1)/episodes)) 
        
            
            #print('callback_1',frame_buffer)
    
  

        
        

        return 'Done Training'   
    return ''

# callbackp para imagenes
@callback(Output('env-plot','figure'),
          [Input('refresher','n_intervals')])
def render(intervals):
    fig=go.Figure()
    
    try:
        img=fsc.get("images")
        fig=px.imshow(img)
        return fig
    except Exception as e:
        #print(e)
        return fig
#callback to load bar

@callback(Output('progressbar', 'value'),
          Output('progressbar-msgs', 'children'),
               [Input('barloader','n_intervals')])
def progress_bar_replay_memory(n_intervals):
    value=fsc1.get("progressbar")
    
    if value is None:
        value='0'
    
    return value,f'{value}%'
#callback to stop training 
@callback(Output('stop-msg','children'),
          [Input('reset-btn','n_clicks'),
           Input('train-btn','n_clicks')])
def stop_job(nclicksreset,nclickscontinue):
    if ctx.triggered_id=='reset-btn':
        fsc2.set('stop','1')
        return 'stopped'
    elif ctx.triggered_id=='train-btn':
        return ''
#calback to status monitoring
@callback(Output('status-msg','children'),
         [Input('statusloader','n_intervals')])
def show_status(status):
    value=fsc3.get('status')
    return value
#callback to plot reward



    

if __name__ == '__main__':
    app.run(debug=True)