# app.py (Fixed Advanced Analytics Section)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import sqlite3
from dotenv import load_dotenv

# Import the updated flight agent
from flight_agent import FlightAnalysisAgent

load_dotenv()

# [Keep all the existing visualization functions as they are]
# ... (create_delay_distribution_chart, create_hourly_analysis, etc.)

def main():
    st.set_page_config(
        page_title="✈️ Mumbai Airport Flight Analysis AI",
        page_icon="✈️",
        layout="wide"
    )
    
    st.title("✈️ Mumbai Airport Flight Scheduling AI")
    st.markdown("**Advanced Flight Operations Analysis** with Interactive Visualizations")
    
    # Initialize agent
    api_key = os.getenv('GEMINI_API_KEY') or st.secrets.get('GEMINI_API_KEY')
    
    if not api_key:
        st.error("❌ Please set GEMINI_API_KEY in your environment or Streamlit secrets")
        st.stop()
    
    # File upload option
    uploaded_file = st.file_uploader(
        "📂 Upload your Flight_Data.xlsx file (optional)", 
        type=['xlsx'], 
        help="Upload the Excel file with flight data."
    )
    
    # Initialize agent
    if 'agent' not in st.session_state:
        try:
            with st.spinner("🔄 Loading flight data and initializing analysis system..."):
                if uploaded_file:
                    st.session_state.agent = FlightAnalysisAgent(api_key, uploaded_file)
                else:
                    st.session_state.agent = FlightAnalysisAgent(api_key)
                    
                st.success("✅ Flight analysis system initialized successfully!")
                
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            st.stop()
    
    # Navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 AI Assistant", 
        "📊 Analytics Dashboard", 
        "📈 Performance Analysis", 
        "🛫 Route & Aircraft Analysis",
        "📉 Advanced Analytics"
    ])
    
    with tab1:
        # AI Assistant Tab (keep existing code)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬 Flight Operations Assistant")
            
            sample_questions = [
                "Which flights have the highest departure delays?",
                "Compare delay patterns between Air India, IndiGo, and Akasa Air",
                "What's the busiest departure time window?",
                "Which routes to Delhi have the most delays?", 
                "Show me performance of different aircraft types",
                "Which flights should be rescheduled to reduce congestion?"
            ]
            
            selected_q = st.selectbox("Choose a sample question:", [""] + sample_questions)
            user_query = st.text_input(
                "Or ask your own question:", 
                value=selected_q,
                placeholder="Ask about flight delays, peak hours, routes, airlines, aircraft types..."
            )
            
            if st.button("🔍 Analyze Flight Data") and user_query:
                with st.chat_message("user"):
                    st.write(user_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing flight operations data..."):
                        try:
                            response = st.session_state.agent.query_flights(user_query)
                            st.write(response)
                        except Exception as e:
                            st.error(f"Query error: {str(e)}")
        
        with col2:
            st.header("🎯 Quick Stats")
            try:
                summary = st.session_state.agent.get_summary_stats()
                if 'error' not in summary:
                    st.metric("Total Flights", summary['total_flights'])
                    st.metric("Avg Delay", f"{summary['avg_delay']} min")
                    
                    if summary.get('airline_performance'):
                        st.subheader("Top Airlines")
                        for airline in summary['airline_performance'][:3]:
                            st.write(f"**{airline['airline']}**: {airline['flights']} flights, {airline['avg_delay']:.1f}min avg delay")
            except Exception as e:
                st.error(f"Stats error: {str(e)}")
    
    with tab2:
        # Analytics Dashboard
        st.header("📊 Flight Operations Dashboard")
        
        try:
            analytics = st.session_state.agent.get_analytics_data()
            
            if 'error' not in analytics:
                # Performance metrics
                if not analytics['correlation_analysis'].empty:
                    st.subheader("⚡ Key Performance Indicators")
                    create_performance_metrics(analytics['correlation_analysis'])
                    st.divider()
                
                # Delay Distribution
                st.subheader("🕒 Delay Distribution Analysis")
                create_delay_distribution_chart(analytics['delay_distribution'])
                
                st.divider()
                
                # Hourly Analysis
                st.subheader("📅 Hourly Departure Patterns")
                create_hourly_analysis(analytics['hourly_departures'])
                
            else:
                st.error(f"Analytics error: {analytics['error']}")
                
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
    
    with tab3:
        # Performance Analysis
        st.header("📈 Performance Analysis")
        
        try:
            analytics = st.session_state.agent.get_analytics_data()
            
            if 'error' not in analytics:
                # Airline Comparison
                st.subheader("🏢 Airline Performance Comparison")
                create_airline_comparison(analytics['airline_comparison'])
                
                st.divider()
                
                # Time Series
                st.subheader("📈 Trend Analysis")
                create_time_series_analysis(analytics['time_series_delays'])
                
            else:
                st.error(f"Performance analysis error: {analytics['error']}")
                
        except Exception as e:
            st.error(f"Performance analysis error: {str(e)}")
    
    with tab4:
        # Route & Aircraft Analysis
        st.header("🛫 Route & Aircraft Analysis")
        
        try:
            analytics = st.session_state.agent.get_analytics_data()
            
            if 'error' not in analytics:
                # Route Performance
                st.subheader("🗺️ Route Performance Analysis")
                create_route_performance_charts(analytics['route_performance'])
                
                st.divider()
                
                # Aircraft Analysis
                st.subheader("✈️ Aircraft Performance Analysis")
                create_aircraft_analysis(analytics['aircraft_analysis'])
                
            else:
                st.error(f"Route & Aircraft analysis error: {analytics['error']}")
                
        except Exception as e:
            st.error(f"Route & Aircraft analysis error: {str(e)}")
    
    with tab5:
        # FIXED Advanced Analytics Tab
        st.header("📉 Advanced Analytics")
        
        try:
            analytics = st.session_state.agent.get_analytics_data()
            
            if 'error' not in analytics:
                # Heatmap
                st.subheader("🔥 Delay Pattern Heatmap")
                create_delay_heatmap(analytics['delay_heatmap'])
                
                st.divider()
                
                # Correlation Analysis
                st.subheader("🔗 Correlation Analysis")
                create_correlation_analysis(analytics['correlation_analysis'])
                
                st.divider()
                
                # FIXED Raw Data Explorer
                st.subheader("🔍 Data Explorer")
                if st.checkbox("Show Raw Flight Data"):
                    try:
                        # FIX: Use direct SQLite connection instead of agent.db.engine
                        conn = sqlite3.connect(st.session_state.agent.db_path)
                        raw_data = pd.read_sql("SELECT * FROM flights LIMIT 100", conn)
                        conn.close()
                        
                        st.dataframe(raw_data, use_container_width=True)
                        
                        # Download option
                        csv = raw_data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Flight Data CSV",
                            data=csv,
                            file_name='mumbai_flights_data.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
                
                # Additional Analytics
                st.subheader("📊 Statistical Summary")
                if not analytics['correlation_analysis'].empty:
                    corr_data = analytics['correlation_analysis']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Departure Delay", f"{corr_data['departure_delay_minutes'].mean():.1f} min")
                        st.metric("Max Departure Delay", f"{corr_data['departure_delay_minutes'].max()} min")
                    
                    with col2:
                        st.metric("Avg Arrival Delay", f"{corr_data['arrival_delay_minutes'].mean():.1f} min")
                        # Calculate delay recovery rate
                        recovery_rate = ((corr_data['departure_delay_minutes'] - corr_data['arrival_delay_minutes']) > 0).mean() * 100
                        st.metric("Delay Recovery Rate", f"{recovery_rate:.1f}%")
                    
                    with col3:
                        st.metric("Peak Hour", f"{corr_data['departure_hour'].mode().iloc[0]}:00")
                        st.metric("Most Used Aircraft", corr_data['aircraft_type'].mode().iloc[0])
                
            else:
                st.error(f"Advanced analytics error: {analytics['error']}")
                
        except Exception as e:
            st.error(f"Advanced analytics error: {str(e)}")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if uploaded_file:
            st.success(f"✅ Using uploaded file: {uploaded_file.name}")
        elif os.path.exists("Flight_Data.xlsx"):
            st.success("✅ Using Flight_Data.xlsx file")
        else:
            st.warning("⚠️ Using fallback sample data")
    
    with col2:
        st.info("**Data Source**: Mumbai Airport July 2025 Operations")
    
    with col3:
        st.info("**Powered by**: Gemini AI + LangChain")

# Add the missing create_performance_metrics function
def create_performance_metrics(data):
    """Create performance metrics dashboard"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        on_time_rate = (data['departure_delay_minutes'] <= 15).mean() * 100
        st.metric(
            "On-Time Rate", 
            f"{on_time_rate:.1f}%",
            f"{'↑' if on_time_rate > 75 else '↓'} vs 75% target"
        )
    
    with col2:
        avg_delay = data['departure_delay_minutes'].mean()
        st.metric(
            "Avg Delay", 
            f"{avg_delay:.1f} min",
            f"{'↓' if avg_delay < 15 else '↑'} vs 15 min target"
        )
    
    with col3:
        worst_delay = data['departure_delay_minutes'].max()
        st.metric("Max Delay", f"{worst_delay} min")
    
    with col4:
        recovery_rate = ((data['departure_delay_minutes'] - data['arrival_delay_minutes']) > 0).mean() * 100
        st.metric(
            "Recovery Rate", 
            f"{recovery_rate:.1f}%",
            "In-flight delay recovery"
        )

# Keep all existing visualization functions (create_delay_distribution_chart, etc.)
def create_delay_distribution_chart(data):
    """Create delay distribution pie and bar chart"""
    col1, col2 = st.columns(2)
    
    with col1:
        fig_pie = px.pie(
            data, 
            values='flights', 
            names='delay_category',
            title="Flight Distribution by Delay Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        fig_bar = px.bar(
            data,
            x='delay_category',
            y='flights',
            title="Number of Flights by Delay Category",
            color='avg_delay',
            color_continuous_scale='Reds',
            text='flights'
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(xaxis_title="Delay Category", yaxis_title="Number of Flights")
        st.plotly_chart(fig_bar, use_container_width=True)

def create_hourly_analysis(data):
    """Create hourly departure analysis"""
    data['hour_label'] = data['hour'] + ':00'
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            x=data['hour_label'],
            y=data['departures'],
            name="Number of Departures",
            marker_color='lightblue',
            yaxis='y'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=data['hour_label'],
            y=data['avg_delay'],
            mode='lines+markers',
            name="Average Delay (min)",
            marker_color='red',
            yaxis='y2'
        )
    )
    
    fig.update_xaxes(title_text="Departure Hour")
    fig.update_yaxes(title_text="Number of Departures", secondary_y=False)
    fig.update_yaxes(title_text="Average Delay (minutes)", secondary_y=True)
    fig.update_layout(title="Hourly Departure Pattern and Delays", height=500)
    
    st.plotly_chart(fig, use_container_width=True)

def create_route_performance_charts(data):
    """Create route performance visualizations"""
    col1, col2 = st.columns(2)
    
    with col1:
        top_delayed = data.nlargest(10, 'avg_delay')
        
        fig1 = px.bar(
            top_delayed,
            x='avg_delay',
            y='route',
            orientation='h',
            title="Top 10 Most Delayed Routes",
            color='flights',
            color_continuous_scale='Reds',
            text='avg_delay'
        )
        fig1.update_traces(texttemplate='%{text:.1f} min', textposition='outside')
        fig1.update_layout(height=500, yaxis_title="Route", xaxis_title="Average Delay (minutes)")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(
            data,
            x='flights',
            y='avg_delay',
            size='worst_delay',
            color='avg_arrival_delay',
            hover_name='route',
            title="Route Performance Analysis",
            labels={
                'flights': 'Number of Flights',
                'avg_delay': 'Average Departure Delay (min)',
                'avg_arrival_delay': 'Avg Arrival Delay'
            },
            color_continuous_scale='RdYlBu_r'
        )
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)

def create_airline_comparison(data):
    """Create airline comparison charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.pie(
            data,
            values='flights',
            names='airline_name',
            title="Market Share by Number of Flights",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            data,
            x='airline_name',
            y=['avg_delay', 'avg_arrival_delay'],
            title="Average Delays by Airline",
            barmode='group'
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

def create_aircraft_analysis(data):
    """Create aircraft performance analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            data,
            x='aircraft_type',
            y='flights',
            title="Aircraft Utilization",
            color='avg_dep_delay',
            color_continuous_scale='Viridis',
            text='flights'
        )
        fig1.update_traces(textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(
            data,
            x='avg_dep_delay',
            y='avg_arr_delay',
            size='flights',
            color='aircraft_type',
            title="Aircraft Delay Performance"
        )
        st.plotly_chart(fig2, use_container_width=True)

def create_delay_heatmap(data):
    """Create delay heatmap"""
    if not data.empty:
        heatmap_data = data.pivot(index='destination', columns='hour', values='avg_delay')
        
        fig = px.imshow(
            heatmap_data.fillna(0),
            labels=dict(x="Departure Hour", y="Destination", color="Avg Delay (min)"),
            title="Delay Patterns: Hour vs Destination Heatmap",
            color_continuous_scale='Reds',
            aspect='auto'
        )
        
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for heatmap visualization.")

def create_time_series_analysis(data):
    """Create time series analysis"""
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.line(
            data,
            x='date',
            y='avg_delay',
            title="Daily Average Delay Trend",
            markers=True,
            color_discrete_sequence=['red']
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            data,
            x='date',
            y='flights',
            title="Daily Flight Volume",
            color='max_delay',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig2, use_container_width=True)

def create_correlation_analysis(data):
    """Create correlation analysis"""
    if not data.empty and len(data) > 1:
        data['departure_hour_num'] = pd.to_numeric(data['departure_hour'], errors='coerce')
        
        numeric_cols = ['departure_delay_minutes', 'arrival_delay_minutes', 'departure_hour_num']
        numeric_data = data[numeric_cols].dropna()
        
        if len(numeric_data) > 1:
            corr_matrix = numeric_data.corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                title="Correlation Matrix: Delays and Departure Time",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            
            for i in range(len(corr_matrix.index)):
                for j in range(len(corr_matrix.columns)):
                    fig.add_annotation(
                        x=j, y=i,
                        text=str(round(corr_matrix.iloc[i, j], 2)),
                        showarrow=False,
                        font=dict(color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
                    )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for correlation analysis.")

if __name__ == "__main__":
    main()
