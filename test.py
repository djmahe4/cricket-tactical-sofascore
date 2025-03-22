#Bowler perspective Analysis

#from defs import *
import datetime,requests
#from defs import get_matches
import http.client
import json
from urllib.parse import urlparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation,PillowWriter
import streamlit as st
import imageio
import tempfile
#from main import app
#from icecream import ic
#import os
#os.environ["PATH"] += os.pathsep + r'C:\ffmpeg-master-latest-win64-gpl\bin'
def reset():
    st.session_state.match_selected = False
    st.session_state.mid = None
    st.session_state.choose_side = None
    st.session_state.players = None
    st.session_state.pid = None
    st.session_state.pname=None
    st.session_state.info=None
    st.session_state.details=None
    st.session_state.p_details=None
    st.session_state.mformat = None
    st.session_state.recent_got = []
    st.session_state.matches = []
    st.session_state.incidents = []
    st.session_state.det = None
    st.session_state.incidents2 = []
    st.session_state.det2 = None
    st.session_state.switch = False
    st.session_state.nmat=None
    st.session_state.h_name = None
    st.session_state.a_name = None
    st.session_state.df=None
    st.session_state.venue = None
    st.session_state.runs=0
    st.session_state.wickets=0
    st.session_state.overs=0
    st.success("Reset Sucesss")
    st.rerun()
    return
def converter(gif_path):
    #os.popen("pip install imageio[ffmpeg]")
    #imageio.plugins.ffmpeg.download()
    try:
        with open(gif_path, 'rb') as f:
            st.write("Found GIF file:", gif_path)
    except FileNotFoundError:
        st.error("GIF file not found!")
        return

    try:
        # Create a temporary file to save the converted video
        with tempfile.NamedTemporaryFile(suffix=".mp4",prefix=gif_path[:-4]) as temp_file:
            gif_reader = imageio.get_reader(gif_path, format='GIF')
            fps = gif_reader.get_meta_data().get('fps', 1)  # Default to 1 FPS if metadata is missing

            #st.write(f"Temporary file created: {temp_file.name}")

            # Create a video writer for MP4 format
            with imageio.get_writer(temp_file.name, format='mp4', fps=fps) as video_writer:
                for frame in gif_reader:
                    video_writer.append_data(frame)

            st.success("Conversion successful!")
            st.video(temp_file.name)

            # Provide a download link for the converted MP4 file
            with open(temp_file.name, "rb") as f:
                st.download_button(label="Download MP4", data=f.read(), file_name=f"{gif_path[:-4]}.mp4", mime="video/mp4")
    except Exception as e:
        st.error(f"An error occurred during conversion: {e}")

def determine_match_format(data):
    """Determines the cricket match format based on innings data.

    Returns:
        "T20", "ODI", "Test", or "Unknown format".
    """
    home_innings = data['homeScore'].get('innings', {})

    away_innings = data['awayScore'].get('innings', {})

    all_innings = list(home_innings.values()) + list(away_innings.values())
    #ic(home_innings,away_innings)
    #print(all_innings,data['id'])
    if not all_innings:
        return "Unknown format"

    total_innings = len(all_innings)
    total_overs = 0

    for innings in all_innings:
        if 'overs' not in innings:  # Handle cases where 'overs' might be missing
            #ic(innings)
            return "Unknown format"
        total_overs += innings['overs']

    # More robust logic based on total overs and number of innings
    #if total_innings > 2: #most likely a test, but check overs
    if total_overs >= 110: #edge case for rain affected test match
        return "Test"
    #else:
        #return "Test"
    #elif total_innings == 4:
        #return "Test"
    elif total_overs <= 50:
        #ic('yes')
        return "T20"
    elif total_overs <= 110:
       return "ODI"
    else:
        return "Unknown format"
def scraper(url):
    #url =
    #ic(url)
    parsed = urlparse(url)
    #conn = http.client.HTTPSConnection(parsed.netloc)
    st.session_state.conn.request("GET", parsed.path)
    res = st.session_state.conn.getresponse()
    data = res.read()
    details = json.loads(data.decode("utf-8"))
    #ic(details.keys())
    return details
def opp_team_venue(mid,pid):
    #if not st.session_state.details:
    details=scraper(f"https://www.sofascore.com/api/v1/event/{mid}")
    st.session_state.details=details
    h_name=st.session_state.details['event']['homeTeam']['name']
    st.session_state.h_name=h_name
    #h_id=details['event']['homeTeam']['id']
    a_name=st.session_state.details['event']['awayTeam']['name']
    st.session_state.a_name=a_name
    #a_id=details['event']['awayTeam']['id']
    venue=st.session_state.details['event']['venue']['name']
    st.session_state.venue=venue
    p_details=scraper(f"https://www.sofascore.com/api/v1/event/{mid}/lineups")
        #st.write(p_details['home']['players'])
    st.session_state.p_details=p_details
    for team in ['home','away']:
        for player in st.session_state.p_details[team]['players']:
            #ic(player['name'])
            if st.session_state.pid==player['player']['id']:
                if team == 'home':
                    st.session_state.h_name=None
                    return
                    #return st.session_state.a_name,st.session_state.venue
                else:
                    st.session_state.a_name=None
                    return
    st.session_state.a_name=''
    st.session_state.h_name=''
def get_matches(pid,matches=[], format="T20", ind=0,mtime=[]):
  mdata = scraper(f"https://www.sofascore.com/api/v1/player/{pid}/events/last/{ind}")
  #ic(mdata.keys())
  #print(jdata['events'][0])
  try:
    for event in mdata['events']:
      ans=determine_match_format(event)
      #ic(ans)
      #print(ans)
      if format == ans:
        print(event["startTimestamp"],event['id'])
        matches.append(event['id'])
        mtime.append(event["startTimestamp"])
        #print(event['id'])
        #ic(event['id'])
  except KeyError:
      combined = list(zip(mtime, matches))
      # Sort in descending order based on timestamp (first element of each pair)
      combined.sort(reverse=True)

      # Unzip back into separate lists
      mtime_sorted, matches_sorted = zip(*combined)

      # Convert back to lists if needed (zip returns tuples)
      # mtime_sorted = list(mtime_sorted)
      matches = list(matches_sorted)
      return matches
  if mdata.get('hasNextPage'):
    get_matches(pid,matches, format, ind + 1,mtime)
  #st.session_state.recent_got.append(matches)
  #print(matches)
  # Combine the lists into pairs and sort
  combined = list(zip(mtime, matches))
  # Sort in descending order based on timestamp (first element of each pair)
  combined.sort(reverse=True)

  # Unzip back into separate lists
  mtime_sorted, matches_sorted = zip(*combined)

  # Convert back to lists if needed (zip returns tuples)
  #mtime_sorted = list(mtime_sorted)
  matches = list(matches_sorted)

  return matches
def analyze_batting_stats(det, batting_type, player_slug):
    """
    Analyzes a bowler's stats against a specific batsman.

    Args:
        det: The dictionary containing batting data.
        batting_type: The key in the dictionary (e.g., 'Right').
        player_slug: The slug of the bowler to analyze (e.g., 'ishant-sharma').

    Returns:
        A Pandas DataFrame with the bowler's stats, or None if no matching data is found.
    """

    if batting_type not in det:
        print(f"batting type '{batting_type}' not found in data.")
        return None

    batting_data = det[batting_type]
    batsmen = batting_data["batsman"]

    # Find indices where the bowler matches the player_slug
    matching_indices = [i for i, batter in enumerate(batsmen) if batter == player_slug]

    if not matching_indices:
        print(f"Batter '{player_slug}' not found in '{batting_type}' data.")
        return None

    # Extract relevant stats for the matching indices
    stats = {key: [value[i] for i in matching_indices] for key, value in batting_data.items()}

    df = pd.DataFrame(stats)

    # Calculate additional stats
    df['dots'] = df['runs'].apply(lambda x: 1 if x == 0 or x== "W" else 0)
    df['is_boundary'] = df['runs'].apply(lambda x: 1 if x in [4, 6] else 0)
    df['balls']=df.shape[0]
    total_runs=df['runs'].apply(lambda x: 0 if x=='W' else x).sum()
    df['total_runs']=total_runs
    df['economy']=df['total_runs']/(df['balls']/6)
    dots=df['dots'].sum()
    df['total_dots']=dots
    df['strike_rate']=(total_runs/df['balls'])*100
    boundaries=df['is_boundary'].sum()
    df['total_boundaries']=boundaries
    dot_percentage=(dots/df['balls'])*100
    boundary_percentage=(boundaries/df['balls'])*100
    df['dot_percentage']=dot_percentage
    df['boundary_percentage']=boundary_percentage
    zone_counts = df['zone'].value_counts().to_dict()
    for zone, count in zone_counts.items():
        df[f'wickets_in_{zone}'] = df.loc[df['zone'] == zone, 'runs'].apply(lambda x: 1 if x == 'W' else 0).sum()
        df[f'runs_in_{zone}'] = df.loc[df['zone']==zone,'runs'].apply(lambda x: 0 if x=='W' else x).sum()
    df['wickets']=df['wicket'].apply(lambda x: 1 if x!='' else 0).sum()
    try:
        df = df.loc[:, df.iloc[-1] != 0 ]
    except Exception as e:
        pd.set_option('display.max_coloumns',None)
        print(e)
    return df
def create_ball_animation(det,role):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    color_map = {0: 'gray', 1: 'blue', 2: 'yellow',3:'brown', 4: 'orange', 6: 'red',"W":'black'}
    circle = Circle((0, 0), radius=5, facecolor='green', edgecolor='black')
    ax1.add_patch(circle)

    # Create a Text object for the title *outside* the update function
    title_text = ax1.text(0.02, 1.05, "", transform=ax1.transAxes, fontsize=12, ha='left', va='top')
    stad_text = ax1.text(0.04, 1.15, "", transform=ax1.transAxes, fontsize=14, ha='left', va='top')
    batters=[]
    def update(frame):
        #ax1.clear()  # No longer needed to clear the whole axes
        #ax1.add_patch(circle) #no longer needed to add the circle again and again

        length = det[role]['length'][frame]
        angle = det[role]['angle'][frame]
        runs = det[role]['runs'][frame]
        batsman_name = det[role]['batsman'][frame]
        batsman_type = role

        x_end = length * np.cos(np.deg2rad(angle))
        y_end = length * np.sin(np.deg2rad(angle))
        color = color_map.get(runs)

        ax1.plot([0, x_end], [0, y_end], color=color, linewidth=2)
        venue_name = det[role]['venue'][frame]
        opp_name = det[role]['opp'][frame]
        stad_text.set_text(f"{venue_name} (vs {opp_name})")
        # Update the text of the title object
        title_text.set_text(f"{batsman_name} ({batsman_type})")
        try:
          if batsman_name != det[role]['batsman'][frame+1] and batsman_name not in batters:
              #ax1.clear()
              df =analyze_batting_stats(det, role, det[role]['batsman'][frame])
              last_row = df.iloc[-1]
              print(f"{batsman_name} ({batsman_type})")
              print(last_row)
              st.markdown(f"## {batsman_name} ({batsman_type})")
              st.dataframe(last_row.transpose())
              batters.append(batsman_name)
        except:
          print("Last record")
        ax1.set_xlim([-10, 10])
        ax1.set_ylim([-10, 10])

        for i, (runs, color) in enumerate(color_map.items()):
            ax1.text(-9.5, 8 - i * 0.7, f'Runs: {runs}', fontsize=12, va='top', color=color)

        # Scatter plot animation (unchanged)
        if frame < len(det[role]['runs']):
            runs_scatter = det[role]['runs'][:frame + 1]
            x_values = det[role]['x'][:frame + 1]
            y_values = det[role]['y'][:frame + 1]
            colors_scatter = [color_map.get(run) for run in runs_scatter]

            ax2.clear()
            ax2.scatter(x_values, y_values, color=colors_scatter)
            #ax2.set_xlim([0, max(det[role]['x']) + 5])
            #ax2.set_ylim([0, max(det[role]['y']) + 5])
            ax2.set_xlim([0, 50])
            ax2.set_ylim([0, 100])
            ax2.invert_yaxis()
            ax2.set_title("Wagon wheel (left) and Pitch plot (right)")

    ani = FuncAnimation(fig, update, frames=len(det[role]['runs']), repeat=False)
    gif_writer = PillowWriter(fps=1)
    ani.save(f'{st.session_state.pname}_bowl_animation_with_{role}.gif', writer=gif_writer)
    converter(f'{st.session_state.pname}_bowl_animation_with_{role}.gif')
    return f'{st.session_state.pname}_bowl_animation_with_{role}.mp4'
def bowler_ball_by_ball(incidents):
    det = {}
    for incident in incidents[::-1]:
        batter_type = incident["batsman"]['cricketPlayerInfo']['batting']

        # Initialize bowler type entry if not already present
        if batter_type not in det:
            det[batter_type] = {
                "runs": [],
                "x": [],
                "y": [],
                "length": [],
                "angle": [],
                "batsman": [],
                "wicket": [],
                "zone": [],
                "opp":[],
                "venue":[]
            }
        # Debugging output
        # print("Bowl Detail:", j.get('bowlDetail'))
        # print("Bowler Slug:", j['bowler'].get('slug', 'No slug'))
        # Append ball data to respective lists
        if "wd" in incident.get("incidentClassLabel").lower():
            print("Wide!")
            continue
        # det[batter_type]["runs"].append(incident['runs'])
        try:
            det[batter_type]["x"].append(incident['ballDetails']['pitchHit']['x'])
            det[batter_type]["y"].append(incident['ballDetails']['pitchHit']['y'])
            det[batter_type]["opp"].append(incident['opp'])
            det[batter_type]["venue"].append(incident['venue'])
            det[batter_type]["length"].append(incident.get('length', 0))
            det[batter_type]["zone"].append(incident.get('zone', ""))
            det[batter_type]["angle"].append(incident.get('angle', 0))
            det[batter_type]['batsman'].append(incident['batsman']['slug'])
            if incident.get('bowlDetail'):
                det[batter_type]["wicket"].append(incident['bowlDetail'])
                det[batter_type]["runs"].append("W")
            else:
                det[batter_type]["wicket"].append("")
                det[batter_type]["runs"].append(incident['runs'])
        except KeyError:
            continue
    return det
#@st.cache_data
def append_ball_data(mid,pid):
    #
    #ic(st.session_state.info)
    #if st.session_state.info is None:
    opp_team_venue(mid,pid)
        #st.session_state.p_details = None
        #st.session_state.details = None
        #ic(info)
        #st.session_state.info=info
    #st.write(st.session_state.info)
    #ic(st.session_state.info)
    try:
        jdata2 = scraper(f"https://www.sofascore.com/api/v1/event/{mid}/incidents")
    except json.JSONDecodeError:
        return
    #incidents = []
    st.session_state.runs = 0
    st.session_state.overs = 0
    st.session_state.wickets = 0
    for i in jdata2['incidents']:
        #ic(i)
        if i["bowler"]["id"] == st.session_state.pid:
            #st.write(i)
            #st.write(info)
            if i['wicket']:
                st.session_state.wickets+=1
            st.session_state.runs += i['runs']
            st.session_state.overs += 1
            if st.session_state.h_name is None:
                i['opp'] = st.session_state.a_name
            else:
                i['opp'] = st.session_state.h_name
            i['venue'] = st.session_state.venue
            #st.write(i)
            #ic(i)
            #st.session_state.info=None
            #incidents.append(i)
            st.session_state.incidents2.append(i)
    if st.session_state.overs > 0:  # Only display if there were any balls faced
        st.write(f"Match {mid}: {st.session_state.incidents2[-1]['opp']} at {st.session_state.incidents2[-1]['venue']}, ",
                 f"{st.session_state.wickets}/{st.session_state.runs}({st.session_state.overs // 6}.{st.session_state.overs % 6})")
    else:
        st.write(f"Match {mid}: No bowling data for player {pid}")

    #return incidents

def bowl():
    # recent = get_matches(786470, format="T20")[:10]
    # incidents = []
    if st.session_state.matches==[]:
        st.warning("Click on Setup option")
        st.error("You havent choosen the player and matches!")
        return
    else:
        if st.button("Reset"):
            reset()
        #st.write(st.session_state.matches)
        if st.session_state.incidents2==[]:
            with st.spinner("Filtering data.."):
                #incidents=[]
                for j in st.session_state.matches:
                    try:
                        st.write(j)
                        #ic(j)
                        append_ball_data(j, st.session_state.pid)

                    except KeyError:
                        continue
        #st.session_state.incidents2 = incidents
        #st.write(st.session_state.incidents2)
        st.success("Filtering Success...")
        #ic(st.session_state.p_details,st.session_state.details)
        #st.write(st.session_state.info)
        #st.session_state.p_details=None
        #st.session_state.details=None
        #st.write(st.session_state)
    #print(st.session_state)
    if st.session_state.incidents2:
        #ic(st.session_state.incidents2)
        with st.spinner("Extracting ball by ball data"):
            det = bowler_ball_by_ball(st.session_state.incidents2)
        st.success("Extraction successful")
        st.session_state.det2 = det
    if st.session_state.det2:
        for role in st.session_state.det2:
            with st.spinner(f"Performance analysis vs {role}"):
                st.markdown(f"# vs {role}")
                create_ball_animation(det, role)
                #video_file = open(file, 'rb')
                #video_bytes = video_file.read()
                #st.video(video_bytes)
                #video_file.close()
        st.success("Process Complete!!")
