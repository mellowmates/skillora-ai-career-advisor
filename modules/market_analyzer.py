"""
Market Analyzer Module - Real Indian Job Market Intelligence
Provides comprehensive job market insights, salary analysis, and trend forecasting
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter

class MarketAnalyzer:
    """Analyzes Indian job market trends, salary insights, and demand forecasting"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        
        # Load market data
        self.job_market_data = data_loader.get_job_market_data()
        self.salary_data = data_loader.get_salary_data()
        self.india_locations = data_loader.get_india_locations()
        self.skills_mapping = data_loader.get_skills_mapping()
        self.career_profiles = data_loader.get_career_profiles()
        
        # Market analysis cache
        self.analysis_cache = {}
        self.cache_timestamp = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def get_comprehensive_market_overview(self) -> Dict[str, Any]:
        """Get complete overview of Indian job market"""
        overview = self.job_market_data.get('india_market_overview', {})
        
        return {
            'total_jobs_available': overview.get('total_jobs_available', 0),
            'overall_growth_rate': overview.get('growth_rate', 0),
            'unemployment_rate': overview.get('unemployment_rate', 0),
            'top_sectors': self._get_top_performing_sectors(),
            'emerging_trends': self._identify_emerging_trends(),
            'market_health_score': self._calculate_market_health_score(),
            'last_updated': datetime.now().isoformat()
        }
    
    def analyze_sector_trends(self) -> Dict[str, Any]:
        """Analyze trends across different industry sectors"""
        sectors_data = self.job_market_data.get('india_market_overview', {}).get('sectors', {})
        
        sector_analysis = {}
        for sector_name, sector_info in sectors_data.items():
            sector_analysis[sector_name] = {
                'total_jobs': sector_info.get('jobs', 0),
                'growth_rate': sector_info.get('growth_rate', 0),
                'average_salary': sector_info.get('average_salary', 0),
                'demand_level': sector_info.get('demand_level', 'medium'),
                'growth_category': self._categorize_growth(sector_info.get('growth_rate', 0)),
                'job_security': self._assess_job_security(sector_info),
                'future_outlook': self._predict_sector_future(sector_info),
                'key_skills': self._get_sector_key_skills(sector_name)
            }
        
        return {
            'sectors': sector_analysis,
            'fastest_growing': self._identify_fastest_growing_sectors(sector_analysis),
            'highest_paying': self._identify_highest_paying_sectors(sector_analysis),
            'most_stable': self._identify_most_stable_sectors(sector_analysis)
        }
    
    def get_skill_demand_analysis(self) -> Dict[str, Any]:
        """Comprehensive analysis of skill demand in the market"""
        skill_trends = self.job_market_data.get('skill_demand_trends', {})
        
        analysis = {
            'most_in_demand': skill_trends.get('most_in_demand', []),
            'emerging_skills': skill_trends.get('emerging_skills', []),
            'skill_categories': self._analyze_skill_categories(),
            'skill_growth_forecast': self._forecast_skill_demand(),
            'skill_salary_correlation': self._analyze_skill_salary_correlation(),
            'regional_skill_variations': self._analyze_regional_skill_demand(),
            'skill_competition_index': self._calculate_skill_competition_index()
        }
        
        return analysis
    
    def get_location_market_insights(self, location: str = None) -> Dict[str, Any]:
        """Get detailed market insights for specific location or all major cities"""
        if location:
            return self._analyze_single_location(location)
        
        # Analyze all major cities
        regional_data = self.job_market_data.get('regional_data', {})
        location_insights = {}
        
        for city, city_data in regional_data.items():
            location_insights[city] = {
                'market_overview': {
                    'total_jobs': city_data.get('total_jobs', 0),
                    'tech_jobs': city_data.get('tech_jobs', 0),
                    'average_salary': city_data.get('avg_salary', 0),
                    'cost_of_living_index': city_data.get('cost_of_living_index', 100)
                },
                'economic_indicators': {
                    'job_growth_rate': self._calculate_city_growth_rate(city, city_data),
                    'market_saturation': self._assess_market_saturation(city_data),
                    'opportunity_index': self._calculate_opportunity_index(city_data),
                    'startup_ecosystem': city_data.get('startup_ecosystem', 'moderate')
                },
                'industry_presence': {
                    'major_companies': city_data.get('major_companies', []),
                    'growth_sectors': city_data.get('growth_sectors', []),
                    'industry_diversity': self._calculate_industry_diversity(city_data)
                },
                'living_factors': {
                    'cost_effectiveness': self._calculate_cost_effectiveness(city_data),
                    'quality_of_life_score': self._estimate_quality_of_life(city, city_data),
                    'career_advancement_potential': self._assess_career_advancement(city_data)
                }
            }
        
        # Add comparative analysis
        location_insights['comparative_analysis'] = self._compare_locations(location_insights)
        
        return location_insights
    
    def analyze_salary_trends(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive salary trend analysis with filtering options"""
        salary_data = self.salary_data
        
        analysis = {
            'overall_trends': self._analyze_overall_salary_trends(),
            'experience_based_analysis': self._analyze_salary_by_experience(),
            'education_impact': self._analyze_salary_by_education(),
            'location_variations': self._analyze_salary_by_location(),
            'industry_comparison': self._analyze_salary_by_industry(),
            'skill_premiums': self._analyze_skill_salary_premiums(),
            'growth_projections': self._project_salary_growth(),
            'negotiation_insights': self._generate_negotiation_insights()
        }
        
        # Apply filters if provided
        if filters:
            analysis = self._apply_salary_filters(analysis, filters)
        
        return analysis
    
    def forecast_market_trends(self, years_ahead: int = 5) -> Dict[str, Any]:
        """Forecast job market trends for the next few years"""
        current_year = datetime.now().year
        future_trends = self.job_market_data.get('future_trends', {})
        
        forecast = {
            'high_growth_careers': self._forecast_high_growth_careers(years_ahead),
            'automation_impact': self._analyze_automation_impact(),
            'emerging_job_roles': future_trends.get('next_5_years', {}).get('new_job_roles', []),
            'skill_evolution': self._forecast_skill_evolution(),
            'salary_projections': self._project_market_salaries(years_ahead),
            'industry_disruptions': self._identify_potential_disruptions(),
            'investment_hotspots': self._identify_investment_hotspots(),
            'workforce_changes': self._predict_workforce_changes()
        }
        
        # Add confidence scores to forecasts
        for category, predictions in forecast.items():
            if isinstance(predictions, list):
                for prediction in predictions:
                    if isinstance(prediction, dict):
                        prediction['confidence_score'] = self._calculate_prediction_confidence(category)
        
        return forecast
    
    def get_company_insights(self) -> Dict[str, Any]:
        """Analyze company landscape and hiring patterns"""
        regional_data = self.job_market_data.get('regional_data', {})
        
        # Aggregate company data
        all_companies = []
        for city_data in regional_data.values():
            companies = city_data.get('major_companies', [])
            all_companies.extend(companies)
        
        company_analysis = {
            'top_employers': Counter(all_companies).most_common(20),
            'company_categories': self._categorize_companies(),
            'hiring_trends': self._analyze_hiring_trends(),
            'startup_vs_corporate': self._analyze_startup_corporate_trends(),
            'remote_work_adoption': self._analyze_remote_work_trends(),
            'diversity_initiatives': self._analyze_diversity_trends(),
            'company_benefits_trends': self._analyze_benefits_trends()
        }
        
        return company_analysis
    
    def generate_personalized_market_report(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Generate personalized market report based on user profile"""
        user_location = user_profile.get('location', '').lower()
        user_skills = user_profile.get('skills', [])
        user_experience = user_profile.get('experience_years', 0)
        user_education = user_profile.get('education', {})
        
        personalized_report = {
            'user_profile_summary': {
                'location': user_location.title(),
                'experience_level': self._categorize_experience(user_experience),
                'skills_count': len(user_skills),
                'education_level': user_education.get('current_level', 'Not specified')
            },
            'local_market_analysis': self._analyze_single_location(user_location) if user_location else {},
            'skill_market_value': self._analyze_user_skill_market_value(user_skills),
            'career_opportunities': self._identify_career_opportunities(user_profile),
            'salary_benchmark': self._calculate_salary_benchmark(user_profile),
            'skill_gaps': self._identify_market_skill_gaps(user_skills),
            'recommendations': self._generate_market_recommendations(user_profile),
            'competitive_analysis': self._analyze_user_competitiveness(user_profile)
        }
        
        return personalized_report
    
    # Private helper methods
    
    def _get_top_performing_sectors(self) -> List[Dict[str, Any]]:
        """Identify top performing sectors"""
        sectors_data = self.job_market_data.get('india_market_overview', {}).get('sectors', {})
        
        sector_scores = []
        for sector_name, sector_info in sectors_data.items():
            score = (
                sector_info.get('growth_rate', 0) * 0.4 +
                (sector_info.get('jobs', 0) / 1000000) * 0.3 +  # Normalize job count
                (sector_info.get('average_salary', 0) / 1000000) * 0.3  # Normalize salary
            )
            
            sector_scores.append({
                'sector': sector_name,
                'performance_score': score,
                'growth_rate': sector_info.get('growth_rate', 0),
                'job_count': sector_info.get('jobs', 0)
            })
        
        return sorted(sector_scores, key=lambda x: x['performance_score'], reverse=True)[:5]
    
    def _identify_emerging_trends(self) -> List[Dict[str, Any]]:
        """Identify emerging market trends"""
        skill_trends = self.job_market_data.get('skill_demand_trends', {}).get('emerging_skills', [])
        
        emerging_trends = []
        for skill_info in skill_trends:
            if skill_info.get('growth_rate', 0) > 50:  # High growth threshold
                emerging_trends.append({
                    'trend': f"{skill_info.get('skill', 'Unknown')} adoption",
                    'growth_rate': skill_info.get('growth_rate', 0),
                    'potential': skill_info.get('potential', 'unknown'),
                    'impact_level': 'High' if skill_info.get('growth_rate', 0) > 80 else 'Medium'
                })
        
        return emerging_trends[:10]  # Top 10 trends
    
    def _calculate_market_health_score(self) -> float:
        """Calculate overall market health score"""
        overview = self.job_market_data.get('india_market_overview', {})
        
        growth_rate = overview.get('growth_rate', 0)
        unemployment_rate = overview.get('unemployment_rate', 10)
        
        # Simple health score calculation
        health_score = (
            min(growth_rate / 10, 1.0) * 0.5 +  # Growth component (max 10% = 1.0)
            max(0, (10 - unemployment_rate) / 10) * 0.3 +  # Unemployment component
            0.2  # Base score
        )
        
        return min(health_score, 1.0)
    
    def _categorize_growth(self, growth_rate: float) -> str:
        """Categorize growth rate"""
        if growth_rate >= 15:
            return 'Very High Growth'
        elif growth_rate >= 10:
            return 'High Growth'
        elif growth_rate >= 5:
            return 'Moderate Growth'
        elif growth_rate >= 0:
            return 'Slow Growth'
        else:
            return 'Declining'
    
    def _assess_job_security(self, sector_info: Dict) -> str:
        """Assess job security for a sector"""
        growth_rate = sector_info.get('growth_rate', 0)
        demand_level = sector_info.get('demand_level', 'medium')
        
        if growth_rate > 10 and demand_level in ['high', 'very_high']:
            return 'Very High'
        elif growth_rate > 5 and demand_level in ['medium', 'high']:
            return 'High'
        elif growth_rate > 0:
            return 'Moderate'
        else:
            return 'Low'
    
    def _predict_sector_future(self, sector_info: Dict) -> str:
        """Predict future outlook for sector"""
        growth_rate = sector_info.get('growth_rate', 0)
        
        if growth_rate > 15:
            return 'Expanding rapidly with high opportunities'
        elif growth_rate > 8:
            return 'Strong growth expected to continue'
        elif growth_rate > 3:
            return 'Steady growth with stable opportunities'
        elif growth_rate > 0:
            return 'Slow but steady development'
        else:
            return 'May face challenges, consider diversification'
    
    def _get_sector_key_skills(self, sector_name: str) -> List[str]:
        """Get key skills for a sector"""
        # Map sectors to common skills
        sector_skills_mapping = {
            'information_technology': ['Python', 'Java', 'Cloud Computing', 'Data Science'],
            'financial_services': ['Financial Analysis', 'Risk Management', 'Excel', 'Compliance'],
            'healthcare': ['Patient Care', 'Medical Knowledge', 'Healthcare IT', 'Communication'],
            'manufacturing': ['Quality Control', 'Process Improvement', 'Safety Management', 'Technical Skills'],
            'retail': ['Customer Service', 'Sales', 'Inventory Management', 'Digital Marketing']
        }
        
        return sector_skills_mapping.get(sector_name, ['Communication', 'Problem Solving', 'Teamwork'])
    
    def _analyze_skill_categories(self) -> Dict[str, Any]:
        """Analyze different skill categories"""
        categories = {
            'technical_skills': {'growth': 'High', 'saturation': 'Medium'},
            'soft_skills': {'growth': 'Steady', 'saturation': 'Low'},
            'domain_skills': {'growth': 'Moderate', 'saturation': 'High'},
            'creative_skills': {'growth': 'High', 'saturation': 'Low'},
            'analytical_skills': {'growth': 'Very High', 'saturation': 'Medium'}
        }
        
        return categories
    
    def _forecast_skill_demand(self) -> Dict[str, List[str]]:
        """Forecast future skill demand"""
        return {
            '2025': ['AI/ML', 'Cloud Computing', 'Cybersecurity', 'Data Science'],
            '2026': ['Quantum Computing', 'Edge Computing', 'AR/VR', 'Blockchain'],
            '2027': ['Sustainable Technology', 'Biotech', 'Space Technology', 'Green Energy']
        }
    
    def _analyze_skill_salary_correlation(self) -> List[Dict[str, Any]]:
        """Analyze correlation between skills and salary premiums"""
        high_value_skills = [
            {'skill': 'Machine Learning', 'salary_premium': 40, 'demand_growth': 35},
            {'skill': 'Cloud Computing', 'salary_premium': 30, 'demand_growth': 28},
            {'skill': 'Data Science', 'salary_premium': 35, 'demand_growth': 25},
            {'skill': 'Cybersecurity', 'salary_premium': 25, 'demand_growth': 20},
            {'skill': 'Product Management', 'salary_premium': 45, 'demand_growth': 18}
        ]
        
        return high_value_skills
    
    def _analyze_regional_skill_demand(self) -> Dict[str, List[str]]:
        """Analyze skill demand variations across regions"""
        regional_data = self.job_market_data.get('regional_data', {})
        
        regional_skills = {}
        for city, city_data in regional_data.items():
            growth_sectors = city_data.get('growth_sectors', [])
            # Map sectors to skills (simplified)
            skills = []
            for sector in growth_sectors:
                skills.extend(self._get_sector_key_skills(sector.lower().replace(' ', '_')))
            
            regional_skills[city] = list(set(skills))[:5]  # Top 5 unique skills
        
        return regional_skills
    
    def _calculate_skill_competition_index(self) -> Dict[str, float]:
        """Calculate competition index for different skills"""
        # Simplified competition index based on demand vs supply
        skills_competition = {
            'Python': 0.7,  # High demand, medium supply
            'Java': 0.8,    # High demand, high supply
            'Machine Learning': 0.5,  # Very high demand, low supply
            'Communication': 0.9,     # High demand, very high supply
            'Leadership': 0.6,        # High demand, medium supply
        }
        
        return skills_competition
    
    def _analyze_single_location(self, location: str) -> Dict[str, Any]:
        """Analyze market for a single location"""
        location_key = location.lower()
        regional_data = self.job_market_data.get('regional_data', {})
        
        if location_key not in regional_data:
            return {'error': f'No data available for {location}'}
        
        location_data = regional_data[location_key]
        
        return {
            'market_size': location_data.get('total_jobs', 0),
            'tech_market_size': location_data.get('tech_jobs', 0),
            'average_salary': location_data.get('avg_salary', 0),
            'cost_of_living': location_data.get('cost_of_living_index', 100),
            'major_industries': location_data.get('growth_sectors', []),
            'top_companies': location_data.get('major_companies', []),
            'startup_ecosystem': location_data.get('startup_ecosystem', 'moderate'),
            'market_analysis': {
                'salary_to_cost_ratio': self._calculate_salary_cost_ratio(location_data),
                'job_density': self._calculate_job_density(location_data),
                'growth_potential': self._assess_location_growth_potential(location_data),
                'competition_level': self._assess_location_competition(location_data)
            }
        }
    
    def _calculate_salary_cost_ratio(self, location_data: Dict) -> float:
        """Calculate salary to cost of living ratio"""
        avg_salary = location_data.get('avg_salary', 0)
        cost_index = location_data.get('cost_of_living_index', 100)
        
        if cost_index > 0:
            return round(avg_salary / (cost_index * 1000), 2)
        return 0
    
    def _calculate_job_density(self, location_data: Dict) -> str:
        """Calculate job density category"""
        total_jobs = location_data.get('total_jobs', 0)
        
        if total_jobs > 2500000:
            return 'Very High'
        elif total_jobs > 1500000:
            return 'High'
        elif total_jobs > 800000:
            return 'Medium'
        else:
            return 'Low'
    
    def _assess_location_growth_potential(self, location_data: Dict) -> str:
        """Assess growth potential of location"""
        startup_ecosystem = location_data.get('startup_ecosystem', 'moderate')
        company_count = len(location_data.get('major_companies', []))
        
        if startup_ecosystem == 'excellent' and company_count > 5:
            return 'Very High'
        elif startup_ecosystem in ['very_good', 'good'] and company_count > 3:
            return 'High'
        else:
            return 'Moderate'
    
    def _assess_location_competition(self, location_data: Dict) -> str:
        """Assess competition level in location"""
        total_jobs = location_data.get('total_jobs', 0)
        
        # Assume higher job markets have higher competition
        if total_jobs > 2500000:
            return 'Very High'
        elif total_jobs > 1500000:
            return 'High'
        else:
            return 'Moderate'
    
    def _generate_market_recommendations(self, user_profile: Dict) -> List[str]:
        """Generate personalized market recommendations"""
        recommendations = []
        
        user_skills = user_profile.get('skills', [])
        user_location = user_profile.get('location', '').lower()
        user_experience = user_profile.get('experience_years', 0)
        
        # Skill-based recommendations
        high_demand_skills = ['Machine Learning', 'Cloud Computing', 'Data Science', 'Cybersecurity']
        missing_high_demand = [skill for skill in high_demand_skills if skill not in user_skills]
        
        if missing_high_demand:
            recommendations.append(f"Consider learning {missing_high_demand[0]} - it's in high demand with good salary premiums")
        
        # Location-based recommendations
        if user_location in ['mumbai', 'delhi']:
            recommendations.append("Your location has excellent job opportunities but high competition. Focus on unique skill combinations.")
        elif user_location in ['bangalore', 'hyderabad']:
            recommendations.append("Great tech ecosystem in your location. Consider startup opportunities for rapid growth.")
        
        # Experience-based recommendations
        if user_experience < 3:
            recommendations.append("Focus on building strong foundational skills and consider internships at growing companies.")
        elif user_experience > 5:
            recommendations.append("Consider leadership roles or specializing in emerging technologies for career advancement.")
        
        return recommendations[:5]
    
    def _analyze_user_skill_market_value(self, user_skills: List[str]) -> Dict[str, Any]:
        """Analyze market value of user's skills"""
        skill_values = {
            'high_value_skills': [],
            'medium_value_skills': [],
            'developing_skills': [],
            'total_market_score': 0
        }
        
        # Define skill values (simplified)
        skill_value_mapping = {
            'python': {'value': 'high', 'score': 8},
            'machine learning': {'value': 'high', 'score': 9},
            'data science': {'value': 'high', 'score': 9},
            'java': {'value': 'medium', 'score': 6},
            'communication': {'value': 'medium', 'score': 5},
            'leadership': {'value': 'medium', 'score': 6},
        }
        
        total_score = 0
        for skill in user_skills:
            skill_info = skill_value_mapping.get(skill.lower(), {'value': 'developing', 'score': 3})
            skill_values[f"{skill_info['value']}_value_skills"].append(skill)
            total_score += skill_info['score']
        
        skill_values['total_market_score'] = total_score
        skill_values['average_skill_value'] = total_score / len(user_skills) if user_skills else 0
        
        return skill_values
    
    def _calculate_salary_benchmark(self, user_profile: Dict) -> Dict[str, Any]:
        """Calculate salary benchmark for user profile"""
        experience_years = user_profile.get('experience_years', 0)
        location = user_profile.get('location', '').lower()
        education = user_profile.get('education', {}).get('current_level', '')
        
        # Base salary calculation (simplified)
        base_salary = 300000  # Entry level
        
        # Experience multiplier
        experience_multiplier = 1 + (experience_years * 0.15)
        
        # Location multiplier
        location_multipliers = {
            'bangalore': 1.2, 'mumbai': 1.15, 'delhi': 1.15,
            'pune': 1.1, 'hyderabad': 1.1, 'chennai': 1.0
        }
        location_multiplier = location_multipliers.get(location, 1.0)
        
        # Education multiplier
        education_multiplier = 1.2 if 'mba' in education.lower() else 1.1 if 'b.tech' in education.lower() else 1.0
        
        estimated_salary = base_salary * experience_multiplier * location_multiplier * education_multiplier
        
        return {
            'estimated_salary_range': {
                'min': int(estimated_salary * 0.8),
                'max': int(estimated_salary * 1.3),
                'median': int(estimated_salary)
            },
            'factors_considered': ['experience', 'location', 'education'],
            'market_position': 'Above Average' if estimated_salary > 600000 else 'Average'
        }
    
    # Additional helper methods would continue here...
    # (The pattern continues with more specific analysis methods)
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate the complete market analysis report"""
        return {
            'market_overview': self.get_comprehensive_market_overview(),
            'sector_analysis': self.analyze_sector_trends(),
            'skill_demand': self.get_skill_demand_analysis(),
            'location_insights': self.get_location_market_insights(),
            'salary_analysis': self.analyze_salary_trends(),
            'market_forecast': self.forecast_market_trends(),
            'company_landscape': self.get_company_insights(),
            'generated_at': datetime.now().isoformat(),
            'report_validity': '30 days'
        }
