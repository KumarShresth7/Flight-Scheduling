import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from dotenv import load_dotenv
from flight_agent import FlightAnalysisAgent

load_dotenv()

class FlightSchedulingApp:
    def __init__(self):
        self.setup_page()
        self.initialize_components()
    
    def setup_page(self):
        st.set_page_config(
            page_title="✈️Flight Scheduling AI",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("✈️Mumbai Airport Flight Scheduling AI")
        st.markdown("""
        **Intelligent Flight Operations Analysis**
        
        Ask questions about:
        - Flight delays and patterns  
        - Peak hour congestion
        - Schedule optimization recommendations
        - Route performance analysis
        """)
    
    def initialize_components(self):
        """Initialize components with error handling"""
        try:
            if 'memory' not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
            
            if 'agent' not in st.session_state:
                api_key = os.getenv('GEMINI_API_KEY')
                if not api_key:
                    st.error("Please set GEMINI_API_KEY in your .env file")
                    st.stop()
                
                with st.spinner("Initializing Flight Analysis Agent..."):
                    st.session_state.agent = FlightAnalysisAgent(api_key)
                    st.success("Flight Analysis Agent initialized successfully!")
        except Exception as e:
            st.error(f"Initialization error: {str(e)}")
            st.stop()
    
    def run(self):
        """Main application interface"""
        with st.sidebar:
            st.header("🎯Quick Queries")
            
            sample_questions = [
                "Which flights have the most departure delays?",
                "What's the busiest departure time window?", 
                "Compare delay patterns by aircraft type",
                "Which flights should be rescheduled to reduce congestion?",
                "Show me flights from BOM to DEL with delays > 15 minutes"
            ]
            
            st.write("**Try these examples:**")
            for i, question in enumerate(sample_questions):
                if st.button(f"📝 {question}", key=f"sample_{i}", use_container_width=True):
                    st.session_state.selected_query = question
            
            st.divider()
            st.header("📊Database Info")
            st.info("""
            **Sample Data Loaded:**
            - 5 Mumbai flights
            - Peak morning hours (6-7 AM)
            - Delay analysis ready
            - Route: BOM → IXC,HYD,DEL,SXR,BLR
            """)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("💬Flight Operations Assistant")
            
            if 'selected_query' in st.session_state:
                user_query = st.session_state.selected_query
                del st.session_state.selected_query
            else:
                user_query = None
            
            if not user_query:
                user_query = st.chat_input("Ask about flight schedules, delays, or optimizations...")
            
            if user_query:
                with st.chat_message("user"):
                    st.write(user_query)
                
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Analyzing flight data..."):
                        try:
                            response = st.session_state.agent.query_flights(user_query)
                            st.write(response)
                            
                            st.session_state.memory.save_context(
                                {"input": user_query},
                                {"output": response}
                            )
                        except Exception as e:
                            st.error(f"❌ Query error: {str(e)}")
                            st.info("💡 Try a simpler question or check your query format.")
        
        with col2:
            st.header("📈 Flight Metrics")
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Avg Delay", "13.0 min", "↓2.5 min")
                st.metric("On-Time %", "60%", "↑10%")
            
            with col_b:
                st.metric("Peak Window", "6:00-6:30 AM", "3 flights")
                st.metric("Worst Route", "BOM-IXC", "20 min avg")
            sample_delays = pd.DataFrame({
                'Flight': ['AI2509', 'AI2625', '6E762', 'QP1891', '6E5352'],
                'Delay_Minutes': [20, 17, 8, 20, 0],
                'Route': ['BOM-IXC', 'BOM-HYD', 'BOM-DEL', 'BOM-SXR', 'BOM-BLR']
            })
            
            fig = px.bar(
                sample_delays, 
                x='Flight', 
                y='Delay_Minutes',
                color='Delay_Minutes',
                title="Departure Delays by Flight",
                color_continuous_scale='Reds',
                labels={'Delay_Minutes': 'Delay (Minutes)', 'Flight': 'Flight Number'}
            )
            
            fig.update_layout(
                xaxis_tickangle=-45,  # Rotate x-axis labels
                showlegend=False,     # Hide color legend for cleaner look
                height=400           # Set chart height
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Route Performance")
            
            # Pie chart for routes
            route_data = sample_delays.groupby('Route')['Delay_Minutes'].mean().reset_index()
            
            fig_pie = px.pie(
                route_data, 
                values='Delay_Minutes', 
                names='Route',
                title="Average Delays by Route"
            )
            fig_pie.update_layout(height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Conversation history
        if st.session_state.memory.buffer:
            with st.expander("💭 Conversation History", expanded=False):
                for message in st.session_state.memory.buffer:
                    if isinstance(message, HumanMessage):
                        st.chat_message("user").write(message.content)
                    elif isinstance(message, AIMessage):
                        st.chat_message("assistant").write(message.content)

def main():
    """Run the Streamlit application"""
    app = FlightSchedulingApp()
    app.run()

if __name__ == "__main__":
    main()
