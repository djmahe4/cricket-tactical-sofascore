import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


# Define color mapping for runs
color_map = {
    0: 'gray',    # No runs
    1: 'blue',    # Single run
    2: 'yellow',  # Two runs
    3: 'violet',
    4: 'orange',  # Four runs
    6: 'red'      # Six runs
}
match_phases = {"Powerplay": range(1,7), "Middle Overs": range(7, 16), "Death Overs": range(16, 21)}

#Batter Analysis
def transform_data(data):
    records = []
    #st.write(data)
    #st.write(data.keys())
    for bowler_type, stats in data.items():
        for i in range(len(stats['runs'])):
            record = {
                'bowler_type': bowler_type,
                'runs': stats['runs'][i] ,#if not type(stats['runs'][i])==str else 0,
                'x': stats['x'][i],
                'y': stats['y'][i],
                'length': stats['length'][i],
                'angle': stats['angle'][i],
                'bowler': stats['bowler'][i],
                'wicket': stats['wicket'][i],
                'zone': stats['zone'][i],
                'opposition': stats['opp'][i],
                'venue': stats['venue'][i],
                'mid':stats['mid'][i]
            }
            records.append(record)
    return pd.DataFrame(records)
def filtered():
    # Convert nested dictionary to DataFrame

    # Load and transform data
    if 'incidents' in st.session_state:
        df = transform_data(st.session_state.det)

        st.title("Batsman Performance Analysis 🔍")

        # Sidebar filters
        st.sidebar.header("Filter Options")

        # Batsman filter (placeholder since data seems focused on one batsman)
        selected_bowlers = st.sidebar.multiselect("Select Bowler", df['bowler'].unique(),
                                                  default=df['bowler'].get(0))

        # Multi-select filters
        selected_opposition = st.sidebar.multiselect("Select Opposition",
                                                     df['opposition'].unique(),
                                                     default=df['opposition'].unique())

        selected_venues = st.sidebar.multiselect("Select Venues",
                                                 df['venue'].unique(),
                                                 default=df['venue'].unique())

        selected_bowler_types = st.sidebar.multiselect("Select Bowler Types",
                                                       df['bowler_type'].unique(),
                                                       default=df['bowler_type'].unique())
        # Batsman filter (placeholder since data seems focused on one batsman)
        selected_runs = st.sidebar.multiselect("Select Runs", df['runs'].unique(),
                                               default=df['runs'].unique())

        # Apply filters
        filtered_df = df[
            (df['opposition'].isin(selected_opposition)) &
            (df['venue'].isin(selected_venues)) &
            (df['bowler_type'].isin(selected_bowler_types)) &
            (df['bowler'].isin(selected_bowlers)) &
            (df['runs'].isin(selected_runs))
            ]

        # Key Metrics
        st.header("Key Performance Indicators 📊")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_runs = filtered_df['runs'].apply(lambda x: 0 if x=='W' else x).sum()
            st.metric("Total Runs", total_runs)

        with col2:
            avg_runs = filtered_df['runs'].apply(lambda x: 0 if x=='W' else x).mean()
            st.metric("Average Runs per Ball", f"{avg_runs:.2f}")

        with col3:
            dismissals = filtered_df[filtered_df['wicket'] != ''].shape[0]
            st.metric("Total Dismissals", dismissals)

        with col4:
            # Create a new column with scaled size values
            #filtered_df['scaled_length'] = filtered_df['length'] / 10
            if len(filtered_df) > 0:
                dismissal_rate = (dismissals / len(filtered_df)) * 100
                st.metric("Dismissal Rate", f"{dismissal_rate:.1f}%")

        # Visualization 1: Ball-by-Ball Performance Map
        st.header("Ball Placement Analysis 🎯")
        fig1 = px.scatter(filtered_df,
                          x='x',
                          y='y',
                          color='runs',
                          size='length',
                          hover_data=['bowler', 'venue', 'opposition'],
                          title="Ball Placement & Outcomes",
                          color_continuous_scale='Viridis')
        # Invert the y-axis
        fig1.update_yaxes(autorange="reversed")

        # Add pitch boundaries
        fig1.add_shape(
            type="rect",
            x0=0, y0=0, x1=50, y1=100,  # Adjust to match your pitch dimensions
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)"
        )

        # Add stumps
        fig1.add_trace(
            go.Scatter(
                x=[0], y=[50],  # Adjust stump positions
                mode="markers+text",
                marker=dict(size=10, color="black"),
                text=["Stumps"],
                textposition="top center"
            )
        )

        # Customize axis labels and range
        fig1.update_layout(
            xaxis_title="Horizontal Position (Pitch Width)",
            yaxis_title="Vertical Position (Pitch Length)",
            xaxis_range=[0, 50],  # Adjust based on your pitch dimensions
            yaxis_range=[0, 50]  # Adjust based on your pitch dimensions
        )
        st.plotly_chart(fig1)
        # Create a Plotly figure
        fig5 = go.Figure()

        # Add a circle to represent the batsman's position
        fig5.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-5, y0=-5, x1=5, y1=5,  # Circle bounds
            line_color="black",
            fillcolor="green",
            opacity=0.5
        )
        # Invert the x-axis
        fig1.update_xaxes(autorange="reversed")
        # Add a circle to represent the batsman's position
        fig5.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-2.5, y0=-2.5, x1=2.5, y1=2.5,  # Circle bounds
            #line_color="white",
            line=dict(
                color='white',  # Line color
                width=2,  # Line width
                dash='dot'  # Dotted line style
            ),
            #fillcolor="green",
            opacity=0.5
        )

        # Loop through each entry in the data
        for length, angle, runs in zip(filtered_df['length'],
                                       filtered_df['angle'],
                                       filtered_df['runs']):
            # Calculate line endpoints
            x_end = -(length * np.cos(np.deg2rad(angle)))  # Convert angle to radians
            y_end = length * np.sin(np.deg2rad(angle))  # Convert angle to radians

            # Determine color based on runs scored
            color = color_map.get(runs, 'black')  # Default to black for unexpected values

            # Add a line for the bat placement
            fig5.add_trace(
                go.Scatter(
                    x=[0, x_end],
                    y=[0, y_end],
                    mode='lines',
                    #line=dict(color=color, width=2),
                    fillcolor=color,
                    showlegend=False
                )
            )

        # Add color mapping as annotations
        for i, (runs, color) in enumerate(color_map.items()):
            fig5.add_annotation(
                x=-9.5,
                y=8 - i * 0.7,
                text=f'Runs: {runs}',
                font=dict(size=12, color=color),
                showarrow=False
            )

        # Customize layout
        fig5.update_layout(
            title="Bat Placement Analysis",
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            xaxis_range=[-10, 10],
            yaxis_range=[-10, 10],
            template="plotly_white"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig5)
        # Visualization 2: Performance by Bowler Type
        st.header("Bowler Type Analysis 🏏")
        col1, col2 = st.columns(2)

        with col1:
            fig2 = px.histogram(filtered_df,
                                x='bowler_type',
                                y='runs',
                                histfunc='sum',
                                title="Total Runs by Bowler Type")
            st.plotly_chart(fig2)

        with col2:
            fig3 = px.pie(filtered_df,
                          names='bowler_type',
                          title="Runs Distribution by Bowler Type",
                          hole=0.3)
            st.plotly_chart(fig3)
        cola,colb=st.columns(2)
        with cola:
            # Filter for dismissals
            wicket_df = filtered_df[filtered_df['wicket'] != '']

            # Create the pie chart
            fig7 = px.pie(
                wicket_df,
                names='bowler_type',  # Group by bowler type
                title="Wicket Distribution by Bowler Type",
                hole=0.3  # Add a hole in the middle for a donut chart
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig7)
        with colb:
            # Group by bowler type and calculate total runs, total balls, and strike rate
            #st.dataframe(filtered_df['runs'])
            bowler_stats_type = filtered_df.groupby('bowler_type').agg(
                total_runs=('runs', lambda x: (x != 'W').sum()),
                total_balls=('runs', 'count'),  # Total balls faced against each bowler type
                strike_rate=('runs', lambda x: ((x != 'W').sum() / x.count()) * 100)  # Strike rate
            ).reset_index()
            # Create the bar chart
            fig8 = px.bar(
                bowler_stats_type,
                x='bowler_type',  # Bowler type on the x-axis
                y='strike_rate',  # Strike rate on the y-axis
                title="Strike Rate by Bowler Type",
                labels={'bowler_type': 'Bowler Type', 'strike_rate': 'Strike Rate'},
                text='strike_rate'  # Display strike rate values on the bars
            )

            # Customize the chart
            fig2.update_traces(texttemplate='%{text:.2f}', textposition='outside')  # Format text
            fig2.update_layout(xaxis_title="Bowler Type", yaxis_title="Strike Rate")
            st.plotly_chart(fig8)

        # Advanced Analysis: Strike Rate and Boundary Percentage
        st.header("Advanced Metrics 📈")

        # Calculate boundary stats
        filtered_df['is_boundary'] = filtered_df['runs'].apply(lambda x: 1 if x in [4, 6] else 0)
        boundaries = filtered_df['is_boundary'].sum()
        boundary_percentage = (boundaries / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0

        # Calculate strike rate
        total_balls = len(filtered_df)
        strike_rate = (total_runs / total_balls) * 100 if total_balls > 0 else 0

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Boundary Percentage", f"{boundary_percentage:.1f}%")
        with col2:
            st.metric("Strike Rate", f"{strike_rate:.1f}")
        # Group by mid and zone
        zone_stats = filtered_df.groupby('zone').agg(
            total_runs=('runs', lambda x: (x != 'W').sum()),  # Total runs in the zone
            total_balls=('runs', 'count'),  # Total balls faced in the zone
            boundaries=('is_boundary', 'sum'),  # Total boundaries in the zone
            matches=('mid', lambda x: x.nunique())  # Number of unique matches in the zone
        ).reset_index()

        # Calculate strike rate
        zone_stats['strike_rate'] = (zone_stats['total_runs'] / zone_stats['total_balls']) * 100

        # Calculate boundary percentage
        zone_stats['boundary_percentage'] = (zone_stats['boundaries'] / zone_stats['total_balls']) * 100

        # Calculate average runs per ball
        zone_stats['average'] = zone_stats['total_runs'] / zone_stats['matches']

        # Sort by mid
        #zone_stats = zone_stats.sort_values(by='mid')
        # Display the results
        st.header("Zone-wise Performance")
        #st.dataframe(zone_stats)
        # Create a bar chart for strike rate by zone and mid
        fig9 = px.bar(
            zone_stats,
            x='zone',
            y='strike_rate',
            color='zone',
            barmode='group',
            hover_data=['boundary_percentage','total_runs'],
            title="Strike Rate by Zone",
            labels={'strike_rate': 'Strike Rate', 'zone': 'Zone', 'mid': 'Match ID'}
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig9)
        # Venue-wise Performance
        st.header("Venue Analysis 🏟️")
        venue_stats = filtered_df.groupby('venue').agg(
            total_runs=('runs', lambda x: (x != 'W').sum()),
            average=('runs', lambda x: (x != 'W').groupby(filtered_df.loc[x.index, 'mid']).sum().mean()),
            strike_rate=('runs', lambda x: ((x != 'W').sum()/x.count()) * 100),
            dismissals=('wicket', lambda x: (x != '').sum())
        ).reset_index()

        fig4 = px.bar(venue_stats,
                      x='venue',
                      y='total_runs',
                      hover_data=['average', 'dismissals','strike_rate'],
                      title="Venue-wise Performance")
        st.plotly_chart(fig4)
        st.header("Vulerability Analysis")
        # Analyze dismissals
        dismissals = filtered_df[filtered_df["wicket"] != ""]
        # Dismissal types
        dismissal_counts = filtered_df[filtered_df["wicket"] != ""]["wicket"].value_counts().reset_index()
        dismissal_counts.columns = ["Dismissal Type", "Count"]
        # Head-to-head analysis
        head_to_head = filtered_df.groupby(["bowler", "bowler_type"]).agg(
            total_runs=("runs", lambda x: (x != 'W').sum()),
            total_balls=("runs", "count"),
            dismissals=("wicket", lambda x: (x != "").sum())
        ).reset_index()

        # Grouped bar chart for head-to-head analysis
        fig14 = px.bar(head_to_head, x="bowler", y="total_runs", color="bowler_type", barmode="group",
                      title="Head-to-Head Performance Against Bowlers",
                      labels={"total_runs": "Total Runs", "bowler": "Bowler"})
        st.plotly_chart(fig14)
        #dismissal_counts = dismissals["wicket"].value_counts()
        st.write("Dismissal Types:\n", dismissal_counts)

        # Most effective bowlers
        effective_bowlers = dismissals["bowler"].value_counts()
        st.write("\nMost Effective Bowlers:\n", effective_bowlers)
        st.header("\nOverall Weaknesses:")
        st.write("- Most effective bowler:", effective_bowlers.idxmax())
        # Raw Data
        st.header("Raw Data 📄")
        st.dataframe(filtered_df)

    else:
        st.warning("No data found in session state!")# Define color mapping for bowling outcomes

# Bowler Analysis
def transform_data2(data):
    records = []
    for batsman_type, stats in data.items():
        for i in range(len(stats['runs'])):
            record = {
                'batter_type': batsman_type,
                'runs_conceded': stats['runs'][i] if stats['runs'][i] != 'W' else 0,
                'wicket': 1 if stats['runs'][i] == 'W' else 0,
                'x': stats['x'][i],
                'y': stats['y'][i],
                'length': stats['length'][i],
                'angle': stats['angle'][i],
                'batsman': stats['batsman'][i],
                'zone':stats['zone'][i],
                'venue': stats['venue'][i],
                'opposition': stats['opp'][i],
                'mid': stats['mid'][i],
                'over':stats['over'][i]
            }
            records.append(record)
    return pd.DataFrame(records)


def filtered2():
    if 'incidents2' in st.session_state:
        df = transform_data2(st.session_state.det2)

        st.title("Bowler Performance Analysis 🔍")

        # Sidebar filters
        st.sidebar.header("Filter Options")

        # Date Range Filter

        # Enhanced Multi-select Filters
        selected_opposition = st.sidebar.multiselect(
            "Select Opposition Teams",
            df['opposition'].unique(),
            default=df['opposition'].unique()
        )

        selected_venues = st.sidebar.multiselect(
            "Select Venues",
            df['venue'].unique(),
            default=df['venue'].unique()
        )

        selected_batsman_types = st.sidebar.multiselect(
            "Select Bowler Types",
            df['batter_type'].unique(),
            default=df['batter_type'].unique()
        )

        # Advanced Filters
        with st.sidebar.expander("Advanced Filters"):
            # Minimum Balls Filter
            min_balls = st.number_input(
                "Minimum Balls Bowled",
                min_value=1,
                max_value=100,
                value=1
            )

            # Phase of Play Filter
            selected_phases = st.multiselect(
                "Match Phases",
                match_phases.keys(),
                default=match_phases.keys()
            )
            # Extract over ranges for selected phases
            selected_over_ranges = []
            for phase in selected_phases:
                selected_over_ranges.extend(match_phases[phase])
        # Apply filters
        filtered_df = df[
            (df['opposition'].isin(selected_opposition)) &
            (df['venue'].isin(selected_venues)) &
            (df['batter_type'].isin(selected_batsman_types)) &
            (df['over'].isin(selected_over_ranges))
            ]


        # Add advanced filtering
        filtered_df = filtered_df.groupby('batsman').filter(lambda x: len(x) >= min_balls)

        # Validation check
        if filtered_df.empty:
            st.warning("No data matches the selected filters!")
            return

        # Enhanced Key Metrics
        st.header("Advanced Bowling KPIs 📊")
        col3, col4 = st.columns(2)

        #with col1:
            #powerplay_eco = calculate_phase_economy(filtered_df, "Powerplay")
            #st.metric("Powerplay Economy", f"{powerplay_eco:.1f}")

        #with col2:
            #death_overs_eco = calculate_phase_economy(filtered_df, "Death Overs")
            #st.metric("Death Overs Economy", f"{death_overs_eco:.1f}")

        with col3:
            top_batsman = get_top_batsmen(filtered_df)
            st.metric(
                "Top Batsman",
                f"{top_batsman[0]}",
                f"Runs Conceded: {top_batsman[1]}"
            )

        with col4:
            best_matchup = get_best_matchup(filtered_df)
            st.metric(
                "Best Matchup",
                f"{best_matchup[0]} ({best_matchup[1]} wkts)",
                f"Dismissal Rate: {best_matchup[2]:.2f}"
            )

        # New Visualization: Phase-wise Performance
        st.header("Phase-wise Analysis 📈")
        # Create line chart
        fig_phase = px.line(
            get_phase_stats(filtered_df),
            x="phase",
            y=["economy_rate", "total_wickets"],  # Use correct column names
            title="Performance Across Match Phases",
            markers=True,
            labels={
                "value": "Metric Value",
                "variable": "Metric",
                "phase": "Match Phase"
            }
        )
        st.plotly_chart(fig_phase)

        # Enhanced Opposition Analysis
        st.header("Opposition Team Analysis 🧑🤝🧑")
        col1, col2 = st.columns(2)

        with col1:
            team_stats = get_team_stats(filtered_df)
            team_stats = team_stats[team_stats['total_wickets'] > 0].dropna()
            fig_team = px.bar(
                team_stats,
                x="opposition",
                y="total_wickets",
                title="Performance Against Teams",
                barmode="group"
            )
            st.plotly_chart(fig_team)

        with col2:
            # Filter out teams with zero wickets
            team_stats=get_team_stats(filtered_df)
            team_stats = team_stats[team_stats['economy_rate'] > 0].dropna()

            # Create bar graph
            fig_team_bar = px.bar(
                team_stats,
                x="opposition",  # Teams on the x-axis
                y="economy_rate",  # Wickets on the y-axis
                title="Runs Distribution vs Teams",
                labels={
                    "opposition": "Opposition Team",
                    "total_wickets": "Total Wickets Taken",
                    "wicket_rate":"Bowler strike rate",
                    "boundary_perecentage":"Boundary %"
                },
                text="opposition"  # Display wicket values on the bars
            )

            # Customize the bar graph
            fig_team_bar.update_traces(
                textposition='outside',  # Position the text above the bars
                marker_color='skyblue'  # Customize bar color
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig_team_bar)

        # Advanced Tableau-style Dashboard
        st.header("Comprehensive Bowling Dashboard 📉")
        filtered_df['is_boundary'] = filtered_df['runs_conceded'].apply(lambda x: 1 if x in [4, 6] else 0)
        # Group by mid and zone
        zone_stats = filtered_df.groupby('zone').agg(
            total_runs=('runs_conceded', lambda x: (x != '').sum()),  # Total runs in the zone
            total_balls=('runs_conceded', 'count'),  # Total balls faced in the zone
            boundaries=('is_boundary', 'sum'),  # Total boundaries in the zone
            matches=('mid', lambda x: x.nunique())  # Number of unique matches in the zone
        ).reset_index()

        # Calculate strike rate
        zone_stats['strike_rate'] = (zone_stats['total_runs'] / zone_stats['total_balls']) * 100

        # Calculate boundary percentage
        zone_stats['boundary_percentage'] = (zone_stats['boundaries'] / zone_stats['total_balls']) * 100

        # Calculate average runs per ball
        zone_stats['average'] = zone_stats['total_runs'] / zone_stats['matches']

        # Sort by mid
        # zone_stats = zone_stats.sort_values(by='mid')
        # Display the results
        st.header("Zone-wise Performance")
        # st.dataframe(zone_stats)
        # Create a bar chart for strike rate by zone and mid
        fig9 = px.bar(
            zone_stats,
            x='zone',
            y='strike_rate',
            color='zone',
            barmode='group',
            hover_data=['boundary_percentage', 'total_runs'],
            title="Strike Rate by Zone",
            labels={'strike_rate': 'Strike Rate', 'zone': 'Zone', 'mid': 'Match ID'}
        )
        st.plotly_chart(fig9)
        # Visualization 1: Ball-by-Ball Performance Map
        st.header("Ball Placement Analysis 🎯")
        fig1 = px.scatter(filtered_df,
                          x='x',
                          y='y',
                          color='runs_conceded',
                          size='length',
                          hover_data=['batsman', 'venue', 'opposition'],
                          title="Ball Placement & Outcomes",
                          color_continuous_scale='Viridis')
        # Invert the y-axis
        fig1.update_yaxes(autorange="reversed")

        # Add pitch boundaries
        fig1.add_shape(
            type="rect",
            x0=0, y0=0, x1=50, y1=100,  # Adjust to match your pitch dimensions
            line=dict(color="black", width=2),
            fillcolor="rgba(0,0,0,0)"
        )

        # Add stumps
        fig1.add_trace(
            go.Scatter(
                x=[0], y=[50],  # Adjust stump positions
                mode="markers+text",
                marker=dict(size=10, color="black"),
                text=["Stumps"],
                textposition="top center"
            )
        )

        # Customize axis labels and range
        fig1.update_layout(
            xaxis_title="Horizontal Position (Pitch Width)",
            yaxis_title="Vertical Position (Pitch Length)",
            xaxis_range=[0, 50],  # Adjust based on your pitch dimensions
            yaxis_range=[0, 50]  # Adjust based on your pitch dimensions
        )
        st.plotly_chart(fig1)
        # Create a Plotly figure
        fig5 = go.Figure()

        # Add a circle to represent the batsman's position
        fig5.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-5, y0=-5, x1=5, y1=5,  # Circle bounds
            line_color="black",
            fillcolor="green",
            opacity=0.5
        )
        # Invert the x-axis
        fig1.update_xaxes(autorange="reversed")
        # Add a circle to represent the batsman's position
        fig5.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=-2.5, y0=-2.5, x1=2.5, y1=2.5,  # Circle bounds
            # line_color="white",
            line=dict(
                color='white',  # Line color
                width=2,  # Line width
                dash='dot'  # Dotted line style
            ),
            # fillcolor="green",
            opacity=0.5
        )

        # Loop through each entry in the data
        for length, angle, runs in zip(filtered_df['length'],
                                       filtered_df['angle'],
                                       filtered_df['runs_conceded']):
            # Calculate line endpoints
            x_end = -(length * np.cos(np.deg2rad(angle)))  # Convert angle to radians
            y_end = length * np.sin(np.deg2rad(angle))  # Convert angle to radians

            # Determine color based on runs scored
            color = color_map.get(runs, 'black')  # Default to black for unexpected values

            # Add a line for the bat placement
            fig5.add_trace(
                go.Scatter(
                    x=[0, x_end],
                    y=[0, y_end],
                    mode='lines',
                    # line=dict(color=color, width=2),
                    fillcolor=color,
                    showlegend=False
                )
            )

        # Add color mapping as annotations
        for i, (runs, color) in enumerate(color_map.items()):
            fig5.add_annotation(
                x=-9.5,
                y=8 - i * 0.7,
                text=f'Runs: {runs}',
                font=dict(size=12, color=color),
                showarrow=False
            )

        # Customize layout
        fig5.update_layout(
            title="Bat Placement Analysis",
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            xaxis_range=[-10, 10],
            yaxis_range=[-10, 10],
            template="plotly_white"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig5)

        # Advanced Data Export
        st.sidebar.header("Data Management")
        if st.sidebar.button("Export to CSV"):
            st.download_button(filtered_df)

    else:
        st.warning("No bowling data available!")


# Helper functions
def calculate_phase_economy(df, phase):
    #st.write(phase)
    # Get the over range for the specified phase
    over_range = match_phases.get(phase, [])

    # Filter the DataFrame for the specified phase
    phase_df = df[df['over'].isin(over_range)]
    if len(phase_df) == 0:
        return 0.0
    return (phase_df['runs_conceded'].sum() /  6)


def get_top_batsmen(df):
    """
    Identifies the batsman who has scored the most runs against the bowler.
    Returns a tuple of (batsman_name, runs_conceded).
    """
    if df.empty:
        return ("No Data", 0)  # Return default values if no data is available

    # Group by batsman and calculate total runs conceded
    top_batsman_stats = df.groupby('batsman')['runs_conceded'].sum().nlargest(1).reset_index()

    # Extract the top batsman's name and runs conceded
    top_batsman = top_batsman_stats.iloc[0]
    return (top_batsman['batsman'], top_batsman['runs_conceded'])


def create_gauge_chart(title, max_value, current_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_value,
        title={'text': title},
        gauge={'axis': {'range': [None, max_value]}}
    ))
    return fig


def get_best_matchup(df):
    """
    Identifies the bowler's best matchup against a batsman.
    Returns a tuple of (batsman_name, wickets_taken, dismissal_rate).
    """
    if df.empty:
        return ("No Data", 0, 0.0)  # Return default values if no data is available

    # Group by batsman and calculate wickets, runs, and balls
    matchup_stats = df.groupby('batsman').agg(
        wickets=('wicket', 'sum'),  # Total wickets taken
        runs_conceded=('runs_conceded', 'sum'),  # Total runs conceded
        balls_bowled=('runs_conceded', 'count')  # Total balls bowled
    ).reset_index()

    # Calculate dismissal rate (wickets per ball)
    matchup_stats['dismissal_rate'] = matchup_stats['wickets'] / matchup_stats['balls_bowled']

    # Find the best matchup (highest dismissal rate)
    best_matchup = matchup_stats.loc[matchup_stats['dismissal_rate'].idxmax()]

    return (best_matchup['batsman'], best_matchup['wickets'], best_matchup['dismissal_rate'])


import plotly.graph_objects as go

def create_comparison_radar(df):
    """
    Creates a radar chart comparing a bowler's performance across key metrics.
    """
    if df.empty:
        return go.Figure()  # Return an empty figure if no data is available

    # Calculate key metrics
    metrics = {
        'Economy Rate': (df['runs_conceded'].sum() / (len(df) / 6)),  # Runs per over
        'Dot Ball %': (len(df[df['runs_conceded'] == 0]) / len(df)) * 100,  # Percentage of dot balls
        'Wicket Rate': (df['wicket'].sum() / len(df)) * 100,  # Wickets per 100 balls
        'Boundary %': (len(df[df['runs_conceded'].isin([4, 6])]) / len(df)) * 100,  # Percentage of boundaries conceded
        'Death Overs Economy': calculate_phase_economy(df, "Death Overs"),  # Economy in death overs
        'Powerplay Economy': calculate_phase_economy(df, "Powerplay")  # Economy in powerplay
    }

    # Convert metrics to a format suitable for radar chart
    categories = list(metrics.keys())
    values = list(metrics.values())

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Bowler Performance'
    ))

    # Customize layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(values) * 1.2]  # Adjust range for better visualization
            )),
        showlegend=True,
        title="Bowler Performance Radar Chart"
    )

    return fig


def create_trend_chart(df):
    #st.write(df)
    """
    Creates a trend chart to show a bowler's performance over time.
    """
    if df.empty:
        return go.Figure()  # Return an empty figure if no data is available

    # Group by match or over (depending on data granularity)
    trend_data = df.groupby('mid').agg(
        economy_rate=('runs_conceded', lambda x: (x.sum() / (len(x) / 6))),  # Economy rate per match
        wickets=('wicket', 'sum'),  # Wickets per match
        dot_balls=('runs_conceded', lambda x: (x == 0).sum())  # Dot balls per match
    ).reset_index()

    # Create trend chart
    fig = go.Figure()

    # Add economy rate trend
    fig.add_trace(go.Scatter(
        x=trend_data['mid'],
        y=trend_data['economy_rate'],
        mode='lines+markers',
        name='Economy Rate'
    ))

    # Add wickets trend
    fig.add_trace(go.Scatter(
        x=trend_data['mid'],
        y=trend_data['wickets'],
        mode='lines+markers',
        name='Wickets',
        yaxis='y2'
    ))

    # Customize layout
    fig.update_layout(
        title="Bowler Performance Over Time",
        xaxis_title="Match ID",
        yaxis_title="Economy Rate",
        yaxis2=dict(
            title="Wickets",
            overlaying='y',
            side='right'
        ),
        showlegend=True
    )

    return


def get_team_stats(df):
    """
    Calculates bowling performance statistics against different opposition teams.
    Returns a DataFrame with team-wise stats.
    """
    if df.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no data is available

    # Group by opposition team and calculate stats
    team_stats = df.groupby('opposition').agg(
        total_runs=('runs_conceded', 'sum'),  # Total runs conceded
        total_wickets=('wicket', 'sum'),  # Total wickets taken
        total_balls=('runs_conceded', 'count'),  # Total balls bowled
        boundaries_conceded=('runs_conceded', lambda x: x.isin([4, 6]).sum())  # Total boundaries conceded
    ).reset_index()

    # Calculate derived metrics
    team_stats['economy_rate'] = (team_stats['total_runs'] / (team_stats['total_balls'] / 6))  # Runs per over
    team_stats['wicket_rate'] = (team_stats['total_wickets'] / team_stats['total_balls']) * 100  # Wickets per 100 balls
    team_stats['boundary_percentage'] = (team_stats['boundaries_conceded'] / team_stats[
        'total_balls']) * 100  # Boundary %
    #st.write(team_stats)
    return team_stats#[team_stats['total_wickets'] > 0]


def get_phase_stats(df):
    """
    Calculates bowling performance statistics across different match phases.
    Returns a DataFrame with phase-wise stats.
    """
    if df.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no data is available

    # Initialize a list to store phase-wise stats
    phase_stats = []

    # Calculate stats for each phase
    for phase, over_range in match_phases.items():
        # Filter data for the current phase
        phase_df = df[df['over'].isin(over_range)]

        # Calculate metrics
        try:
            total_runs = phase_df['runs_conceded'].sum()
            boundaries= phase_df['runs_conceded'].apply(lambda x: 1 if x==(4 or 6) else 0).sum()
        except:
            total_runs = phase_df['runs'].apply(lambda x: 0 if x=='W' else x).sum()
            boundaries=phase_df['runs'].apply(lambda x: 1 if x==(4 or 6) else 0).sum()
        total_wickets = phase_df['wicket'].sum()
        total_balls = len(phase_df)
        economy_rate = (total_runs / (total_balls / 6)) if total_balls > 0 else 0
        strike_rate = (total_runs / total_balls)* 100 if total_balls > 0 else 0
        wicket_rate = (total_wickets / total_balls * 100) if total_balls > 0 else 0

        # Append phase stats to the list
        phase_stats.append({
            'phase': phase,
            'total_runs': total_runs,
            'total_wickets': total_wickets,
            'total_balls': total_balls,
            'economy_rate': economy_rate,
            'strike_rate': strike_rate,
            'boundary%':(boundaries/total_balls)*100,
            'wicket_rate': wicket_rate
        })
    #st.write(phase_stats)

    # Convert the list to a DataFrame
    return pd.DataFrame(phase_stats)


def get_venue_stats(df):
    """
    Calculates bowling performance statistics for each venue.
    Returns a DataFrame with venue-wise stats.
    """
    if df.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no data is available

    # Group by venue and calculate stats
    venue_stats = df.groupby('venue').agg(
        total_runs=('runs_conceded', 'sum'),  # Total runs conceded
        total_wickets=('wicket', 'sum'),  # Total wickets taken
        total_balls=('runs_conceded', 'count'),  # Total balls bowled
        boundaries_conceded=('runs_conceded', lambda x: x.isin([4, 6]).sum())  # Total boundaries conceded
    ).reset_index()

    # Calculate derived metrics
    venue_stats['economy_rate'] = (venue_stats['total_runs'] / (venue_stats['total_balls'] / 6))  # Runs per over
    venue_stats['wicket_rate'] = (venue_stats['total_wickets'] / venue_stats[
        'total_balls']) * 100  # Wickets per 100 balls
    venue_stats['boundary_percentage'] = (venue_stats['boundaries_conceded'] / venue_stats[
        'total_balls']) * 100  # Boundary %

    return venue_stats