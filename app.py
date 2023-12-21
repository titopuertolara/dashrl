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
import pandas as pd

def moving_average(x, span=100):
    return pd.DataFrame({'x': np.asarray(x)}).x.ewm(span=span).mean().values


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server
fsc = FileSystemCache("cache_dir")# show frames
fsc1 = FileSystemCache("cache_dir")# progressbar
fsc2 = FileSystemCache("cache_dir")# stop flag
fsc3 = FileSystemCache("cache_dir")# status warning
fsc4 = FileSystemCache("cache_dir")# plot reward
fsc5 = FileSystemCache("cache_dir") # text training progress

suggested_params={
    'CartPole':{'episodes':200,'batch_size':32,'memory_size':500,'gamma':0.95,'epsilon':1,'decrement':0.995,'lr':0.001},
    'LunarLander':{'episodes':2000,'batch_size':64,'memory_size':10000,'gamma':0.995,'epsilon':1,'decrement':0.0005,'lr':0.001}
}

fsc1.set('progressbar','0')
zeros_img=np.full((400,600,3),255,dtype=np.uint8)
fsc.set('images',zeros_img)
fsc2.set('stop','0')
fsc3.set('status','idle')
fsc4.set('plotreward',[])
fsc5.set('textparams',{'episode':0,'reward':0,'epsilon':0,'steps':0})
app.layout = html.Div([
    html.H4('Deep Reinforcement learning simulator'),
    html.Div(dcc.Dropdown(id='models-option',
        options=[
            {'label':'Lunar Lander','value':'LunarLander'},
            {'label':'Cart Pole','value':'CartPole'}],
            value='CartPole'
    )),
    html.Div([
            html.Div(id='discount-factor-div',children=[html.P('Gamma'),dcc.Input(id='discount_factor_input',type='text')]),
            html.Div(id='epsilon-factor-div',children=[html.P('Epsilon'),dcc.Input(id='epsilon_input',type='text')]),
            html.Div(id='epsilon-discount-div',children=[html.P('Decrement'),dcc.Input(id='epsilon_decrement_factor',type='text')]),
            html.Div(id='learning-rate-div',children=[html.P('Learning rate'),dcc.Input(id='learning_rate',type='text')]),
            html.Div(id='memory-size-div',children=[html.P('Replay memory size'),dcc.Input(id='memory_size',type='text')]),
            html.Div(id='episodes-size-div',children=[html.P('Episodes'),dcc.Input(id='episode_size',type='text')]),
            html.Div(id='batch-size-div',children=[html.P('Batch size'),dcc.Input(id='batch_size',type='text')])



        ]),
    html.Div([
                
        html.Div(id='env-div',children=[dcc.Graph(id='env-plot')],style={'display':'inline-block','width':'40%'}),
        
        html.Div(id='reward-div-plot',children=[dcc.Graph(id='reward-plot')],style={'display':'inline-block','width':'40%'}),
        html.Div(id='text-div',
            children=[
                html.Div(id='episode-div'),
                html.Div(id='reward-div'),
                html.Div(id='epsilon-div'),
                html.Div(id='steps-div')


            ],style={'display':'inline-block','width':'20%'}
        
        )
    ]),
    html.Div(html.Progress(id='progressbar',value='0',max='100')),
    html.Div(id='progressbar-msgs'),
    html.Div(html.Button('Train',id='train-btn',n_clicks=0)),
    html.Div(html.Button('Stop',id='reset-btn',n_clicks=0)),
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
            memory_size=10000
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
                    fsc4.set('plotreward',total_rewards)
                    fsc5.set('textparams',{'episode':e,'reward':rewards,'epsilon':agent.epsilon,'steps':steps})
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
def show_status(statusintervals):
    value=fsc3.get('status')
    return value
#callback to plot reward
@callback(Output('reward-plot','figure'),
          [Input('statusloader','n_intervals')])
def show_reward(plotintervals):
    total_rewards=fsc4.get('plotreward')
    fig=go.Figure()
    fig.add_trace(go.Scatter(y=total_rewards,mode='lines+markers',name='Total reward'))
    fig.add_trace(go.Scatter(y=moving_average(total_rewards),name='Mean reward'))
    fig.update_xaxes(title='Episodes')
    fig.update_yaxes(title='Reward')
    return fig
#callback to monitoring parameters
@callback(Output('episode-div','children'),
          Output('reward-div','children'),
          Output('epsilon-div','children'),
          Output('steps-div','children'),
         [Input('statusloader','n_intervals')])
def show_parameters(parameters_interval):
    vals=fsc5.get('textparams')
    episode=vals['episode']
    rewards=vals['reward']
    epsilon=vals['epsilon']
    steps=vals['steps']
    return f'Episode : {episode}',f'Reward: {rewards}',f'Epsilon: {epsilon}',f'Steps: {steps}'




    

if __name__ == '__main__':
    app.run(debug=True)