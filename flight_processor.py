import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document
import re
from dotenv import load_dotenv
import os
load_dotenv()


gemini_api_key = os.getenv("GEMINI_API_KEY")

class FlightDataProcessor:
    def __init__(self, gemini_api_key):
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_output_tokens=8192
        )
        
    def load_and_process_data(self, file_path):
        """Load Excel data and process with LangChain"""
        # Load Excel file
        df_6am = pd.read_excel(file_path, sheet_name='6AM - 9AM', skiprows=1)
        df_9am = pd.read_excel(file_path, sheet_name='9AM - 12PM', skiprows=1)
        
        # Combine datasets
        combined_df = pd.concat([df_6am, df_9am], ignore_index=True)
        
        # Clean and structure data using Gemini
        cleaned_data = self._clean_data_with_gemini(combined_df)
        
        return cleaned_data
    
    def _clean_data_with_gemini(self, df):
        """Use Gemini to intelligently clean and structure flight data"""
        
        cleaning_prompt = PromptTemplate(
            input_variables=["raw_data"],
            template="""
            You are an expert aviation data analyst. Clean and structure this flight data:

            Raw Data Sample:
            {raw_data}

            Requirements:
            1. Extract flight_number, date, route (from-to), aircraft_type
            2. Convert STD/ATD to proper datetime, calculate departure_delay_minutes
            3. Convert STA/ATA to proper datetime, calculate arrival_delay_minutes  
            4. Parse status from text like "Landed 8:14 AM" to extract actual times
            5. Handle missing data appropriately
            6. Identify peak departure time windows (5-minute buckets)

            Return structured JSON with fields:
            - flight_number, date, origin, destination, aircraft_type
            - scheduled_departure, actual_departure, departure_delay_minutes
            - scheduled_arrival, actual_arrival, arrival_delay_minutes
            - status, peak_time_bucket

            Focus on Mumbai (BOM) departures only.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=cleaning_prompt)
        
        # Process data in chunks
        processed_flights = []
        for chunk in np.array_split(df, 10):  # Process in smaller chunks
            chunk_str = chunk.to_string()
            result = chain.run(raw_data=chunk_str)
            processed_flights.append(result)
            
        return processed_flights

    def calculate_delay_metrics(self, processed_data):
        """Calculate comprehensive delay analytics"""
        
        analytics_prompt = PromptTemplate(
            input_variables=["flight_data"],
            template="""
            Analyze flight delay patterns for Mumbai Airport scheduling optimization:

            Data: {flight_data}

            Calculate:
            1. Average departure delays by time slot (6-9AM, 9AM-12PM)
            2. Peak congestion windows (identify 5-minute periods with most departures)
            3. Route-specific delay patterns (which destinations have consistent delays)
            4. Aircraft type performance (A20N vs A21N vs B38M delay patterns)
            5. Cascade delay impact (flights delayed due to previous flight delays)
            6. Weather/operational delay indicators

            Provide actionable recommendations for:
            - Schedule optimization (which flights to move by ±10 minutes)
            - Runway capacity management
            - Delay risk scoring for future flights

            Format as structured analysis with specific recommendations.
            """
        )
        
        chain = LLMChain(llm=self.llm, prompt=analytics_prompt)
        analysis = chain.run(flight_data=str(processed_data))
        
        return analysis


    
        