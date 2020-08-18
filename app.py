import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np

import plotly
import plotly.graph_objects as go

#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import date, timedelta
import datetime

import fbprophet
from fbprophet import Prophet

app = dash.Dash(__name__)
server = app.server
app.config.suppress_callback_exceptions = True
app.title = 'COVID-19'

data = pd.read_csv('data/dashboard_data.csv')
data['date'] = pd.to_datetime(data['date'])

# selects the "data last updated" date
update = data['date'].dt.strftime('%B %d, %Y').iloc[-1]

dash_colors = {
    'background': '#E7EDEF',
    'text': '#0F0C0C',
    'grid': '#333333',
    'red': '#BF0000',
    'blue': '#466fc2',
    'green': '#5bc246',
    'green_rc':'#4ECB35'
}

available_countries = sorted(data['Country/Region'].unique())

states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
          'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
          'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
          'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
          'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
          'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
          'New Jersey', 'New Mexico', 'New York', 'North Carolina',
          'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
          'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
          'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
          'West Virginia', 'Wisconsin', 'Wyoming']


region_options = {'United States': states}



df_us = pd.read_csv('data/df_us.csv')
df_us['percentage'] = df_us['percentage'].astype(str)


df_us_counties = pd.read_csv('data/df_us_county.csv')
df_us_counties['percentage'] = df_us_counties['percentage'].astype(str)
df_us_counties['Country/Region'] = df_us_counties['Country/Region'].astype(str)

# Model


@app.callback(
    Output('confirmed_ind', 'figure'),
    [Input('global_format', 'value')])
def confirmed(view):
    '''
    creates the CUMULATIVE CONFIRMED indicator
    '''
    if view == 'United States':
        df = df_us
    
    value = df[df['date'] == df['date'].iloc[-1]]['Confirmed'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Confirmed'].sum()
    return {
            'data': [{'type': 'indicator',
                    'mode': 'number+delta',
                    'value': value,
                    'delta': {'reference': delta,
                              'valueformat': ',g',
                              'relative': False,
                              'increasing': {'color': dash_colors['blue']},
                              'decreasing': {'color': dash_colors['green']},
                              'font': {'size': 25}},
                    'number': {'valueformat': ',',
                              'font': {'size': 50}},
                    'domain': {'y': [0, 1], 'x': [0, 1]}}],
            'layout': go.Layout(
                title={'text': "CUMULATIVE CONFIRMED"},
                font=dict(color=dash_colors['red']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                height=200
                )
            }

@app.callback(
    Output('active_ind', 'figure'),
    [Input('global_format', 'value')])
def active(view):
    '''
    creates the CURRENTLY ACTIVE indicator
    '''
    if view == 'United States':
        df = df_us
    
    value = df[df['date'] == df['date'].iloc[-1]]['Active'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Active'].sum()
    return {
            'data': [{'type': 'indicator',
                    'mode': 'number+delta',
                    'value': value,
                    'delta': {'reference': delta,
                              'valueformat': ',g',
                              'relative': False,
                              'increasing': {'color': dash_colors['blue']},
                              'decreasing': {'color': dash_colors['green']},
                              'font': {'size': 25}},
                    'number': {'valueformat': ',',
                              'font': {'size': 50}},
                    'domain': {'y': [0, 1], 'x': [0, 1]}}],
            'layout': go.Layout(
                title={'text': "CURRENTLY ACTIVE"},
                font=dict(color=dash_colors['red']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                height=200
                )
            }

@app.callback(
    Output('recovered_ind', 'figure'),
    [Input('global_format', 'value')])
def recovered(view):
    '''
    creates the RECOVERED CASES indicator
    '''
    if view == 'United States':
        df = df_us

    value = df[df['date'] == df['date'].iloc[-1]]['Recovered'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Recovered'].sum()
    return {
            'data': [{'type': 'indicator',
                    'mode': 'number+delta',
                    'value': value,
                    'delta': {'reference': delta,
                              'valueformat': ',g',
                              'relative': False,
                              'increasing': {'color': dash_colors['blue']},
                              'decreasing': {'color': dash_colors['green']},
                              'font': {'size': 25}},
                    'number': {'valueformat': ',',
                              'font': {'size': 50}},
                    'domain': {'y': [0, 1], 'x': [0, 1]}}],
            'layout': go.Layout(
                title={'text': "RECOVERED CASES"},
                font=dict(color=dash_colors['green_rc']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                height=200
                )
            }

@app.callback(
    Output('deaths_ind', 'figure'),
    [Input('global_format', 'value')])
def deaths(view):
    '''
    creates the DEATHS TO DATE indicator
    '''
    if view == 'United States':
        df = df_us

    value = df[df['date'] == df['date'].iloc[-1]]['Deaths'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Deaths'].sum()
    return {
            'data': [{'type': 'indicator',
                    'mode': 'number+delta',
                    'value': value,
                    'delta': {'reference': delta,
                              'valueformat': ',g',
                              'relative': False,
                              'increasing': {'color': dash_colors['blue']},
                              'decreasing': {'color': dash_colors['green']},
                              'font': {'size': 25}},
                    'number': {'valueformat': ',',
                              'font': {'size': 50}},
                    'domain': {'y': [0, 1], 'x': [0, 1]}}],
            'layout': go.Layout(
                title={'text': "DEATHS TO DATE"},
                font=dict(color=dash_colors['red']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                height=200
                )
            }

@app.callback(
    Output('us_trend', 'figure'),
    [Input('global_format', 'value'),
     Input('population_select', 'value')])
def us_trend(view, population):
    '''
    creates the upper-left chart (aggregated stats for the view)
    '''
    
    if view == 'United States':
        df = df_us
        df_us.loc[df_us['Country/Region'] == 'Recovered', ['population']] = 0
    

    if population == 'absolute':
        confirmed = df.groupby('date')['Confirmed'].sum()
        active = df.groupby('date')['Active'].sum()
        recovered = df.groupby('date')['Recovered'].sum()
        deaths = df.groupby('date')['Deaths'].sum()
        title_suffix = ''
        hover = '%{y:,g}'
    elif population == 'percent':
        df = df.dropna(subset=['population'])
        confirmed = df.groupby('date')['Confirmed'].sum() / df.groupby('date')['population'].sum()
        active = df.groupby('date')['Active'].sum() / df.groupby('date')['population'].sum()
        recovered = df.groupby('date')['Recovered'].sum() / df.groupby('date')['population'].sum()
        deaths = df.groupby('date')['Deaths'].sum() / df.groupby('date')['population'].sum()
        title_suffix = ' per 100,000 people'
        hover = '%{y:,.2f}'
    else:
        confirmed = df.groupby('date')['Confirmed'].sum()
        active = df.groupby('date')['Active'].sum()
        recovered = df.groupby('date')['Recovered'].sum()
        deaths = df.groupby('date')['Deaths'].sum()
        title_suffix = ''
        hover = '%{y:,g}'

    traces = [go.Scatter(
                    x=df.groupby('date')['date'].first(),
                    y=confirmed,
                    hovertemplate=hover,
                    name="Confirmed",
                    mode='lines'),
                go.Scatter(
                    x=df.groupby('date')['date'].first(),
                    y=active,
                    hovertemplate=hover,
                    name="Active",
                    mode='lines'),
                go.Scatter(
                    x=df.groupby('date')['date'].first(),
                    y=recovered,
                    hovertemplate=hover,
                    name="Recovered",
                    mode='lines'),
                go.Scatter(
                    x=df.groupby('date')['date'].first(),
                    y=deaths,
                    hovertemplate=hover,
                    name="Deaths",
                    mode='lines')]
    return {
            'data': traces,
            'layout': go.Layout(
                title="{} Infections{}".format(view, title_suffix),
                xaxis_title="Date",
                yaxis_title="Number of Cases",
                font=dict(color=dash_colors['text']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                xaxis=dict(gridcolor=dash_colors['grid']),
                yaxis=dict(gridcolor=dash_colors['grid'])
                )
            }

@app.callback(
    Output('country_select', 'options'),
    [Input('global_format', 'value')])
def set_active_options(selected_view):
    '''
    sets allowable options for regions in the upper-right chart drop-down
    '''
    return [{'label': i, 'value': i} for i in region_options[selected_view]]

@app.callback(
    Output('country_select', 'value'),
    [Input('global_format', 'value'),
     Input('country_select', 'options')])
def set_countries_value(view, available_options):
    '''
    sets default selections for regions in the upper-right chart drop-down
    '''
    
    if view == 'United States':
        return ['New York', 'New Jersey', 'California', 'Texas', 'Florida', 'Georgia', 'Arizona', 'North Carolina', 'Oklahoma']
    

@app.callback(
    Output('active_states', 'figure'),
    [Input('global_format', 'value'),
     Input('country_select', 'value'),
     Input('column_select', 'value'),
     Input('population_select', 'value')])
def active_countries(view, countries, column, population):
    '''
    creates the upper-right chart (sub-region analysis)
    '''
   
    if view == 'United States':
        df = df_us
    

    if population == 'absolute':
        column_label = column
        hover = '%{y:,g}<br>%{x}'
    elif population == 'percent':
        column_label = '{} per 100,000'.format(column)
        df = df.dropna(subset=['population'])
        hover = '%{y:,.2f}<br>%{x}'
    else:
        column_label = column
        hover = '%{y:,g}<br>%{x}'

    traces = []
    countries = df[(df['Country/Region'].isin(countries)) &
                   (df['date'] == df['date'].max())].groupby('Country/Region')['Active'].sum().sort_values(ascending=False).index.to_list()
    for country in countries:
        if population == 'absolute':
            y_data = df[df['Country/Region'] == country].groupby('date')[column].sum()
            recovered = df[df['Country/Region'] == 'Recovered'].groupby('date')[column].sum()
        elif population == 'percent':
            y_data = df[df['Country/Region'] == country].groupby('date')[column].sum() / df[df['Country/Region'] == country].groupby('date')['population'].first()
            recovered = df[df['Country/Region'] == 'Recovered'].groupby('date')[column].sum() / df[df['Country/Region'] == country].groupby('date')['population'].first()
        else:
            y_data = df[df['Country/Region'] == country].groupby('date')[column].sum()
            recovered = df[df['Country/Region'] == 'Recovered'].groupby('date')[column].sum()

        traces.append(go.Scatter(
                    x=df[df['Country/Region'] == country].groupby('date')['date'].first(),
                    y=y_data,
                    hovertemplate=hover,
                    name=country,
                    mode='lines'))
    if column == 'Recovered':
        traces.append(go.Scatter(
                    x=df[df['Country/Region'] == 'Recovered'].groupby('date')['date'].first(),
                    y=recovered,
                    hovertemplate=hover,
                    name='Unidentified',
                    mode='lines'))
    return {
            'data': traces,
            'layout': go.Layout(
                    title="{} by Region".format(column_label),
                    xaxis_title="Date",
                    yaxis_title="Number of Cases",
                    font=dict(color=dash_colors['text']),
                    paper_bgcolor=dash_colors['background'],
                    plot_bgcolor=dash_colors['background'],
                    xaxis=dict(gridcolor=dash_colors['grid']),
                    yaxis=dict(gridcolor=dash_colors['grid']),
                    hovermode='closest'
                )
            }

@app.callback(
    Output('US_map', 'figure'),
    [Input('global_format', 'value'),
     Input('date_slider', 'value')])
def world_map(view, date_index):
    '''
    creates the lower-left chart (map)
    '''
   
    if view == 'United States':
        scope = 'usa'
        projection_type = 'albers usa'
        df = df_us_counties
        sizeref = 7
   
    df = df[(df['date'] == df['date'].unique()[date_index]) & (df['Confirmed'] > 0)]
    return {
            'data': [
                go.Scattergeo(
                    lon = df['Longitude'],
                    lat = df['Latitude'],
                    text = df['Country/Region'] + ': ' +\
                        ['{:,}'.format(i) for i in df['Confirmed']] +\
                        ' total cases, ' + df['percentage'] +\
                        '% from previous week',
                    hoverinfo = 'text',
                    mode = 'markers',
                    marker = dict(reversescale = True,
                        autocolorscale = False,
                        symbol = 'circle',
                        size = np.sqrt(df['Confirmed']),
                        sizeref = sizeref,
                        sizemin = 0,
                        line = dict(width=1, color='rgba(102, 102, 102)'),
                        colorscale = 'Blues',
                        cmin = 0,
                        color = df['share_of_last_week'],
                        cmax = 100,
                        colorbar = dict(
                            title = "Percentage of <br> cases",
                            thickness = 30)
                        )
                    )
            ],
            'layout': go.Layout(
                title ='County_Wise Spread<br>Over the time period',
                geo=dict(scope=scope,
                        projection_type=projection_type,
                        showland = True,
                        landcolor = "rgb(229, 229, 229)",
                        showocean = True,
                        oceancolor = "rgb(80, 150, 250)",
                        showcountries=True,
                        showlakes=True),
                font=dict(color=dash_colors['text']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background']
            )
        }

def hex_to_rgba(h, alpha=1):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])

@app.callback(
    Output('trajectory', 'figure'),
    [Input('global_format', 'value'),
     Input('date_slider', 'value')])
def trajectory(view, date_index):
    '''
    creates the lower-right chart (trajectory)
    '''
    
    if view == 'United States':
        df = data[data['Country/Region'] == 'US']
        df = df.drop('Country/Region', axis=1)
        df = df.rename(columns={'Province/State': 'Country/Region'})
        scope = 'states'
        threshold = 1000
    

    date = data['date'].unique()[date_index]

    df = df.groupby(['date', 'Country/Region'], as_index=False)['Confirmed'].sum()
    df['previous_week'] = df.groupby(['Country/Region'])['Confirmed'].shift(7, fill_value=0)
    df['new_cases'] = df['Confirmed'] - df['previous_week']
    df['new_cases'] = df['new_cases'].clip(lower=0)

    xmax = np.log(1.25 * df['Confirmed'].max()) / np.log(10)
    xmin = np.log(threshold) / np.log(10)
    ymax = np.log(1.25 * df['new_cases'].max()) / np.log(10)
    ymin = np.log(10)

    countries_full = df.groupby(by='Country/Region', as_index=False)['Confirmed'].max().sort_values(by='Confirmed', ascending=False)['Country/Region'].to_list()
    
    df = df[df['date'] <= date]

    countries = df.groupby(by='Country/Region', as_index=False)['Confirmed'].max().sort_values(by='Confirmed', ascending=False)
    countries = countries[countries['Confirmed'] > threshold]['Country/Region'].to_list()
    countries = [country for country in countries_full if country in countries]

    traces = []
    trace_colors = plotly.colors.qualitative.D3
    color_idx = 0

    for country in countries:
        filtered_df = df[df['Country/Region'] == country].reset_index()
        idx = filtered_df['Confirmed'].sub(threshold).gt(0).idxmax()
        trace_data = filtered_df[idx:].copy()
        trace_data['date'] = pd.to_datetime(trace_data['date'])
        trace_data['date'] = trace_data['date'].dt.strftime('%b %d, %Y')

        marker_size = [0] * (len(trace_data) - 1) + [10]
        color = trace_colors[color_idx % len(trace_colors)]
        marker_color = 'rgba' + str(hex_to_rgba(color, 1))
        line_color = 'rgba' + str(hex_to_rgba(color, .5))

        traces.append(
            go.Scatter(
                    x=trace_data['Confirmed'],
                    y=trace_data['new_cases'],
                    mode='lines+markers',
                    marker=dict(color=marker_color,
                                size=marker_size,
                                line=dict(width=0)),
                    line=dict(color=line_color, width=2),
                    name=country,
                    text = ['{}: {:,} confirmed; {:,} from previous week'.format(country,
                                                                                trace_data['Confirmed'].iloc[i],
                                                                                trace_data['new_cases'].iloc[i]) \
                                                                                    for i in range(len(trace_data))],
                    hoverinfo='text')
        )

        color_idx += 1

    return {
        'data': traces,
        'layout': go.Layout(
                title='Statewise spread over the time'.format(scope, threshold),
                xaxis_type="log",
                yaxis_type="log",
                xaxis_title='Total Confirmed Cases',
                yaxis_title='New Confirmed Cases (in the past week)',
                font=dict(color=dash_colors['text']),
                paper_bgcolor=dash_colors['background'],
                plot_bgcolor=dash_colors['background'],
                xaxis=dict(gridcolor=dash_colors['grid'],
                           range=[xmin, xmax]),
                yaxis=dict(gridcolor=dash_colors['grid'],
                           range=[ymin, ymax]),
                hovermode='closest',
                showlegend=True
            )
        }


@app.callback(Output('preds', 'figure'),
              [Input('my-dropdowntest', "value"), Input("radiopred", "value")])
def update_graph(state, radioval):
    dropdown = {"New York": "Newyork","California": "California","Colorado": "Colorado","Washington": "Washington","Alaska": "Alaska","Arizona": "Arizona","Arkansas": "Arkansas","Connecticut": "Connecticut","Delaware": "Delaware","District of Columbia": "District of Columbia","Florida": "Florida","Georgia": "Georgia","Hawaii": "Hawaii","Texas": "Texas","Illinois": "Illinois","New Jersey": "New Jersey","New Mexico": "New Mexico","Virginia": "Virginia","Indiana": "Indiana","Ohio": "Ohio",}
    radio = {"Confirmed": "Total Cases", "Recovered": "Recovery", "Deaths": "Deaths", }
    trace1 = []
    trace2 = []
    trace3 = []
    trace4 = []
    
    


    if (state == None):
        trace1.append(
            go.Scatter(x= [0], y= [0],
                       mode='markers', opacity=0.7, textposition='bottom center'))
        traces = [trace1]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
                  'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                      height=600, title=f"{radio[radioval]}",
                                      paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')}
    else:

        df = df_us[(df_us['Country/Region'] == state)]

        Dates = df['date']
        Total_Cases = df[radioval]

        Day = df.groupby(['date'])[radioval].sum()
        '''x = np.arange(len(Day))
        y = Day.values
        days = x.reshape(-1,1)

        # Future forecasting for the next 30 days

        days_in_future = 30
        future_forecast = np.array([i for i in range(len(Day)+days_in_future)]).reshape(-1,1)
        adjusted_dates = future_forecast[:-30]

        # Converting all integers to datetime for better visuals
        start = '2020-1-22'
        start_date= datetime.datetime.strptime(start,'%Y-%m-%d')
        future_dates = []


        for i in range(len(future_forecast)):
          future_dates.append((start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))

        Poly = PolynomialFeatures(degree=4)
        X = Poly.fit_transform(days)

        reg = LinearRegression()
        reg.fit(X,y)
        y_Predict_train = reg.predict(X)

        poly_pred=reg.predict(Poly.transform(future_forecast))'''

        # Facebook Prophet Model

        global_cases = Day.reset_index()
        Total_cases = global_cases[["date",radioval]]
        Total_cases.rename(columns={"date":"ds",radioval:"y"},inplace=True)
        train = Total_cases

        m= Prophet()
        m.add_seasonality(name="monthly",period=30.5,fourier_order=5)
        # Fit Model
        m.fit(train)
        # Future Date
        future_dates = m.make_future_dataframe(periods=60)

        # Prediction
        prediction =  m.predict(future_dates)

        

        '''trace1.append(go.Scatter(x=Dates,y=Total_Cases, mode='lines',
            opacity=0.7,name=f'Actual Data',textposition='bottom center',marker = dict(color = 'rgba(16, 112, 2, 0.8)')))
        trace2.append(go.Scatter(x=future_dates,y=poly_pred,mode='lines',line=dict(color='royalblue', width=4, dash='dot'),marker = dict(color = 'rgba(80, 26, 80, 0.8)'),
            opacity=0.6,name=f'Predicted Data',textposition='bottom center'))'''



        trace1.append(go.Scatter( name = 'Live Data',
                                mode = 'lines+markers',
                                x = list(train['ds']),
                                y = list(train['y']),
                                marker=dict(
                                color='#FFBAD2',
                                line=dict(width=1))))

        trace2.append(go.Scatter(
    name = 'trend',
    mode = 'lines',
    x = list(prediction['ds']),
    y = list(prediction['yhat']),
    marker=dict(
        color='red',
        line=dict(width=3)
    )
))

        trace3.append(go.Scatter(
    name = 'upper band',
    mode = 'lines',
    x = list(prediction['ds']),
    y = list(prediction['yhat_upper']),
    line= dict(color='#57b88f'),
    fill = 'tonexty',
    fillcolor='rgba(26,150,65,0.15)'
))

        trace4.append(go.Scatter(
    name= 'lower band',
    mode = 'lines',
    x = list(prediction['ds']),
    y = list(prediction['yhat_lower']),
    line= dict(color='#1705ff')
))


        traces = [trace1, trace2, trace4, trace3]
        data = [val for sublist in traces for val in sublist]
        figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                height=600,title=f"{radio[radioval]} Over the time in {dropdown[state]}",
                xaxis={"title":"Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},yaxis={"title":f"No of {radio[radioval]} Cases"},     paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)')}
    return figure


app.layout  = html.Div([ html.H1(children='Welcome to COVID-19 Dashboard USA',
        style={
            
            'textAlign': 'center',
            'color': dash_colors['text']
            }
        ),

    html.Div(children='Data last updated {} end-of-day'.format(update), style={
        'textAlign': 'center',
        'color': dash_colors['text']
        }),

    html.Div(dcc.RadioItems(id='global_format',
            options=[{'label': i, 'value': i} for i in ['United States']],
            value='United States',
            labelStyle={'float': 'center', 'display': 'inline-block'}
            ), style={'textAlign': 'center',
                'color': dash_colors['text'],
                'width': '100%',
                'float': 'center',
                'display': 'inline-block'
            }
        )  ,
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Live', children=[
    html.Div(style={'backgroundColor': dash_colors['background']}, children=[
   

    html.Div(dcc.Graph(id='confirmed_ind'),
        style={
            'textAlign': 'center',
            'color': dash_colors['red'],
            'width': '25%',
            'float': 'left',
            'display': 'block'
            }
        ),

    html.Div(dcc.Graph(id='active_ind'),
        style={
            'textAlign': 'center',
            'color': dash_colors['red'],
            'width': '25%',
            'float': 'left',
            'display': 'block'
            }
        ),

    html.Div(dcc.Graph(id='deaths_ind'),
        style={
            'textAlign': 'center',
            'color': dash_colors['red'],
            'width': '25%',
            'float': 'left',
            'display': 'inline-block'
            }
        ),

    html.Div(dcc.Graph(id='recovered_ind'),
        style={
            'textAlign': 'center',
            'color': dash_colors['red'],
            'width': '25%',
            'float': 'left',
            'display': 'inline-block'
            }
        ),

    html.Div(dcc.Markdown('Display data in the below two charts as total values or as values relative to population:'),
        style={
            'textAlign': 'center',
            'color': dash_colors['text'],
            'width': '100%',
            'float': 'center',
            'display': 'inline-block'}),

    html.Div(dcc.RadioItems(id='population_select',
            options=[{'label': 'Total values', 'value': 'absolute'},
                        {'label': 'Values per 100,000 of population', 'value': 'percent'}],
            value='absolute',
            labelStyle={'float': 'center', 'display': 'inline-block'},
            style={'textAlign': 'center',
                'color': dash_colors['text'],
                'width': '100%',
                'float': 'center',
                'display': 'inline-block'
                })
        ),

    html.Div(  # us_trend and active_states
        [
            html.Div(
                dcc.Graph(id='us_trend'),
                style={'width': '50%', 'float': 'left', 'display': 'inline-block'}
                ),
            html.Div([
                dcc.Graph(id='active_states'),
                html.Div([
                    dcc.RadioItems(
                        id='column_select',
                        options=[{'label': i, 'value': i} for i in ['Confirmed', 'Active', 'Recovered', 'Deaths']],
                        value='Active',
                        labelStyle={'float': 'center', 'display': 'inline-block'},
                        style={'textAlign': 'center',
                            'color': dash_colors['text'],
                            'width': '100%',
                            'float': 'center',
                            'display': 'inline-block'
                            }),
                    dcc.Dropdown(
                        id='country_select',
                        multi=True,
                        style={'width': '95%', 'float': 'center'}
                        )],
                    style={'width': '100%', 'float': 'center', 'display': 'inline-block'})
                ],
                style={'width': '50%', 'float': 'right', 'vertical-align': 'bottom'}
            )],
        style={'width': '98%', 'float': 'center', 'vertical-align': 'bottom'}
        ),

    html.Div(dcc.Markdown(' '),
        style={
            'textAlign': 'center',
            'color': dash_colors['text'],
            'width': '100%',
            'float': 'center',
            'display': 'inline-block'}),

    html.Div(dcc.Graph(id='US_map'),
        style={'width': '50%',
            'display': 'inline-block'}
        ),

    html.Div([dcc.Graph(id='trajectory')],
        style={'width': '50%',
            'float': 'left',
            'display': 'inline-block'}),

    html.Div(html.Div(dcc.Slider(id='date_slider',
                min=list(range(len(data['date'].unique())))[0],
                max=list(range(len(data['date'].unique())))[-1],
                value=list(range(len(data['date'].unique())))[-1],
                marks={(idx): {'label': date.format(u"\u2011", u"\u2011") if
                    (idx-4)%7==0 else '', 'style':{'transform': 'rotate(30deg) translate(0px, 7px)'}} for idx, date in
                    enumerate(sorted(set([item.strftime("%m{}%d{}%Y") for
                    item in data['date']])))},  # for weekly marks,
                # marks={(idx): (date.format(u"\u2011", u"\u2011") if
                #     date[4:6] in ['01', '15'] else '') for idx, date in
                #     enumerate(sorted(set([item.strftime("%m{}%d{}%Y") for
                #     item in data['date']])))},  # for bi-monthly makrs
                step=1,
                vertical=False,
                updatemode='mouseup'),
            style={'width': '94.74%', 'float': 'left'}),  # width = 1 - (100 - x) / x
        style={'width': '95%', 'float': 'right'}),  # width = x
    
    html.Div(dcc.Markdown('''
            &nbsp;  
            &nbsp;  


             
            Source data: [Johns Hopkins CSSE](https://github.com/CSSEGISandData/COVID-19)  
              
            '''),
            style={
                'textAlign': 'center',
                'color': dash_colors['text'],
                'width': '100%',
                'float': 'center',
                'display': 'inline-block'}
            )
        ])
,
]), 
dcc.Tab(label='Forecast', children=[
html.Div([html.H1("Machine Learning", style={"textAlign": "center"}), html.H2("Model Forecast", style={"textAlign": "left"}),
    dcc.Dropdown(id='my-dropdowntest',value="New York",options=[{'label': 'Newyork', 'value': 'New York'},{'label': 'California', 'value': 'California'},{'label': 'Colorado', 'value': 'Colorado'},{'label': 'Washington', 'value': 'Washington'},{'label': 'Alaska', 'value': 'Alaska'},{'label': 'Arizona', 'value': 'Arizona'},{'label': 'Arkansas', 'value': 'Arkansas'},{'label': 'Connecticut', 'value': 'Connecticut'},{'label': 'Delaware', 'value': 'Delaware'},{'label': 'District of Columbia', 'value': 'District of Columbia'},{'label': 'Florida', 'value': 'Florida'},{'label': 'Georgia', 'value': 'Georgia'},{'label': 'Hawaii', 'value': 'Hawaii'},{'label': 'Texas', 'value': 'Texas'},{'label': 'Illinois', 'value': 'Illinois'},{'label': 'New Jersey', 'value': 'New Jersey'},{'label': 'New Mexico', 'value': 'New Mexico'},{'label': 'Virginia', 'value': 'Virginia'},{'label': 'Indiana', 'value': 'Indiana'},{'label': 'Ohio', 'value': 'Ohio'}],
                style={"display": "block", "margin-left": "auto", "margin-right": "auto", "width": "50%"}),
          dcc.RadioItems(id="radiopred", value="Confirmed", labelStyle={'display': 'inline-block', 'padding': 10},
                         options=[{'label': "Total Cases", 'value': "Confirmed"}, {'label': "Recovery", 'value': "Recovered"},
                                  {'label': "Deaths", 'value': "Deaths"}], style={'textAlign': "center", }),
 
    dcc.Graph(id='preds'), 
],)
], className="container")
])
])


if __name__ == '__main__':
    app.run_server(debug=True)
