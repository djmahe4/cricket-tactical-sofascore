import streamlit as st
import datetime
from defs import *
import requests
from test import bowl
#from icecream import ic

@st.cache_resource
def conn_make():
    url = f"https://www.sofascore.com/api/v1/event/12527965"
    parsed = urlparse(url)
    conn = http.client.HTTPSConnection(parsed.netloc)
    st.session_state.conn=conn
    #conn.request("GET", parsed.path)

def app():
    st.title("Sofascore Tactical Analysis")
    st.write(datetime.datetime.today())

    st.session_state.match_selected = True

    # Show dropdowns only after the "Start Analysis" button is clicked
    if st.session_state.match_selected and not st.session_state.choose_side:
        choice = st.selectbox("Match", list(st.session_state.choices.keys()))
        match_id = st.session_state.choices[choice]
        st.write(f"Selected match: {choice}")
        st.write(f"Match ID: {match_id}")
        st.session_state.mid=match_id
    if st.session_state.mid and not st.session_state.choose_side:
        side = st.radio("Home/Away", ["homeTeam", "awayTeam"])
        if st.button("Show Players"):
            st.session_state.choose_side = side
    if st.session_state.choose_side and not st.session_state.pid:
        tid=requests.get(f"https://www.sofascore.com/api/v1/event/{st.session_state.mid}").json()['event'][st.session_state.choose_side]['id']
        players={x["player"]['name']:x["player"]['id'] for x in requests.get(f"https://www.sofascore.com/api/v1/team/{tid}/players").json()['players']}
        st.session_state.players=players
        player = st.selectbox("Players", list(st.session_state.players.keys()))
        pid = st.session_state.players[player]
        if st.button("Choose"):
            st.session_state.pid=pid
            st.session_state.pname=player
            st.write(pid,player)
    if st.session_state.pid and not st.session_state.mformat:
        mformat=st.selectbox("Choose format",["T20","ODI","Test"])
        if st.button("Select"):
            st.session_state.mformat=mformat
            #st.write(st.session_state)
    if st.session_state.mformat and st.session_state.recent_got == []:
        with st.spinner("Getting recent data.."):
            recent=get_matches(st.session_state.pid,format=st.session_state.mformat)
        st.success("Got recent stats")
        #ic(st.session_state.recent_got)
        st.session_state.recent_got=recent
    if 'recent_got' in st.session_state and st.session_state.recent_got != []:
        #nmat=st.slider("Select number of matches",min_value=0,max_value=len(st.session_state.recent_got))
        st.success(f"Found data for {len(st.session_state.recent_got)} {st.session_state.mformat} matches of {st.session_state.pname}")
        nmat=st.number_input(f"Enter an integer less than {len(st.session_state.recent_got)}")
        st.session_state.nmat=nmat
        st.write(st.session_state.nmat)
        if st.button("Done"):
            st.session_state.matches=st.session_state.recent_got[:int(st.session_state.nmat)]
            st.write(st.session_state.matches)
            st.success("Great now choose any of Them!")
            ba,bo=st.columns(2)
            with ba:
                st.page_link(st.Page(bat))
            with bo:
                st.page_link(st.Page(bowl))
def bat():
    # incidents=[]
    if st.session_state.matches==[]:
        st.error("You havent choosen the player and matches!")
        st.page_link(st.Page(app))
        return
    else:
        if st.button("Reset"):
            reset()
        with st.spinner("Filtering data.."):
            st.write(st.session_state.matches)
            if st.session_state.incidents==[]:
                for j in st.session_state.matches:
                    try:
                        append_bat_data(j, st.session_state.pid)
                        # st.session_state.incidents = incidents
                    except KeyError as e:
                        st.write(e)
                        continue
        st.success("Filtering Success...")
        #st.write(st.session_state.incidents)

        if st.session_state.incidents :
            with st.spinner("Extracting ball by ball data"):
                det=batter_ball_by_ball(st.session_state.incidents)
            st.success("Extraction successful")
            st.session_state.det=det
        if st.session_state.det :
            for role in st.session_state.det:
                with st.spinner(f"Performance analysis vs {role}"):
                    st.markdown(f"# vs {role}")
                    create_bat_animation(det,role)
                    #video_file = open(file, 'rb')
                    #video_bytes = video_file.read()
                    #st.video(video_bytes)
                    #video_file.close()
            st.success("Process Complete!!")
if __name__=="__main__":
    # Define a button to start the analysis after choices are made
    if 'match_selected' not in st.session_state:
        conn_make()
        contents = init()  # Only calls match_id_init once
        choices = contents

        st.session_state.choices = choices
        st.session_state.match_selected = False
        st.session_state.mid = None
        st.session_state.info=None
        st.session_state.choose_side = None
        st.session_state.players = None
        st.session_state.pid = None
        st.session_state.pname=None
        st.session_state.details = None
        st.session_state.p_details = None
        st.session_state.h_name=None
        st.session_state.a_name=None
        st.session_state.venue=None
        st.session_state.mformat = None
        st.session_state.df=None
        st.session_state.recent_got = []
        st.session_state.matches = []
        st.session_state.incidents = []
        st.session_state.det = None
        st.session_state.incidents2 = []
        st.session_state.det2 = None
        st.session_state.switch = False
        st.session_state.runs=0
        st.session_state.overs=0
        st.session_state.balls=0
        st.session_state.wickets=0
    pg=st.navigation([st.Page(app,title="Setup"),st.Page(bat,title="Batting"),st.Page(bowl,title="Bowling"),
                      st.Page(reset,title='Reset'),st.Page(expansion,title="Shot Notations")])
    pg.run()
