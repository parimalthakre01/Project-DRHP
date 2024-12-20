import json
import PyPDF2
import numpy as np
import requests
from typing import List, Dict, Any
from datetime import datetime

class ICDRComplianceChecker:
    def __init__(self, icdr_pdf_path: str, api_token: str):
        """
        Initialize the compliance checker with ICDR regulations
        
        Args:
            icdr_pdf_path (str): Path to ICDR 2018 PDF
            api_token (str): Nugen API authentication token
        """
        self.api_token = api_token
        self.icdr_regulations = {}
        self.regulation_embeddings = None
        
        # API endpoints
        self.embedding_url = "https://api.nugen.in/inference/embeddings"
        self.completion_url = "https://api.nugen.in/inference/completions"
        
        # Headers for API calls
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        # Load and process ICDR regulations
        self.load_icdr_regulations(icdr_pdf_path)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embeddings using Nugen API"""
        payload = {
            "input": text,
            "model": "nugen-flash-embed",
            "dimensions": 123
        }
    
        try:
            response = requests.post(
                self.embedding_url, 
                json=payload, 
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
        
        # Extract the embedding from the 'data' field
            if 'data' in data and isinstance(data['data'], list) and 'embedding' in data['data'][0]:
                embedding = data['data'][0]['embedding']
                return np.array(embedding)
            else:
                raise Exception(f"Missing 'embedding' in API response: {data}")
        except Exception as e:
            raise Exception(f"Embedding API error: {str(e)}")


    
    def get_completion(self, prompt: str) -> str:
        """Get completion using Nugen API"""
        payload = {
            "max_tokens": "2000",
            "model": "nugen-flash-instruct",
            "prompt": prompt,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(
                self.completion_url, 
                json=payload, 
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()['choices'][0]['text']
        except Exception as e:
            raise Exception(f"Completion API error: {str(e)}")
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF document"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                full_text = ""
                for page in reader.pages:
                    full_text += page.extract_text()
                return full_text
        except Exception as e:
            raise Exception(f"PDF extraction error: {str(e)}")
    
    def split_into_chapters(self, text: str) -> List[str]:
        """Split ICDR regulations into chapters/sections"""
        # This needs to be customized based on ICDR PDF structure
        chapters = []
        current_chapter = ""
        
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith('CHAPTER') or line.strip().startswith('SCHEDULE'):
                if current_chapter:
                    chapters.append(current_chapter)
                current_chapter = line
            else:
                current_chapter += "\n" + line
        
        if current_chapter:
            chapters.append(current_chapter)
        
        return chapters
    
    def load_icdr_regulations(self, icdr_path: str):
        """Load and process ICDR regulations"""
        try:
            # Extract text from ICDR PDF
            icdr_text = self.extract_pdf_text(icdr_path)
            
            # Split into chapters/sections
            chapters = self.split_into_chapters(icdr_text)
            
            # Store chapters
            self.icdr_regulations = {
                f'chapter_{i}': chapter for i, chapter in enumerate(chapters)
            }
            
            # Create embeddings for chapters
            chapter_embeddings = []
            for chapter in chapters:
                embedding = self.get_embedding(chapter)
                chapter_embeddings.append(embedding)
            
            self.regulation_embeddings = np.array(chapter_embeddings)
            
        except Exception as e:
            raise Exception(f"Error processing ICDR regulations: {str(e)}")
    
    def analyze_compliance(self, prospectus_section: str, regulation_section: str) -> Dict:
        """Analyze compliance of prospectus section with regulation"""
        prompt = f"""
        Analyze the compliance of this prospectus section:
        {prospectus_section}
        
        With respect to this SEBI ICDR 2018 regulation:
        {regulation_section}
        
        Provide:
        1. Specific compliance issues identified
        2. Required modifications for compliance
        3. Severity of non-compliance (Low/Medium/High)
        4. Relevant ICDR regulation references
        """
        
        analysis = self.get_completion(prompt)
        return {
            'prospectus_section': prospectus_section,
            'regulation_reference': regulation_section,
            'analysis': analysis
        }
    
    def check_prospectus_compliance(self, prospectus_path: str) -> Dict[str, Any]:
        """Check prospectus compliance against ICDR regulations"""
        # Process prospectus
        prospectus_text = self.extract_pdf_text(prospectus_path)
        prospectus_sections = self.split_into_sections(prospectus_text)
        
        compliance_results = []
        
        # Analyze each section
        for section in prospectus_sections:
            section_embedding = self.get_embedding(section)
            
            # Find most relevant regulation section
            similarities = np.dot(self.regulation_embeddings, section_embedding)
            most_relevant_idx = np.argmax(similarities)
            similarity_score = similarities[most_relevant_idx]
            
            # If potentially non-compliant, analyze in detail
            if similarity_score < 0.6:
                analysis = self.analyze_compliance(
                    section,
                    self.icdr_regulations[f'chapter_{most_relevant_idx}']
                )
                compliance_results.append({
                    'section': section,
                    'similarity_score': float(similarity_score),
                    'compliance_status': 'Non-Compliant',
                    'analysis': analysis
                })
            else:
                compliance_results.append({
                    'section': section,
                    'similarity_score': float(similarity_score),
                    'compliance_status': 'Compliant'
                })
        
        return self.generate_report(compliance_results)
    
    def split_into_sections(self, text: str) -> List[str]:
        """Split prospectus into analyzable sections"""
        # This needs to be customized based on prospectus structure
        # For now, using simple paragraph splitting
        sections = []
        current_section = ""
        
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(current_section.split()) > 500:
                sections.append(current_section)
                current_section = para
            else:
                current_section += "\n" + para
        
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def generate_report(self, compliance_results: List[Dict]) -> Dict:
        """Generate comprehensive compliance report"""
        non_compliant_count = sum(
            1 for result in compliance_results 
            if result['compliance_status'] == 'Non-Compliant'
        )
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_sections': len(compliance_results),
                'compliant_sections': len(compliance_results) - non_compliant_count,
                'non_compliant_sections': non_compliant_count,
                'overall_compliance_percentage': (
                    (len(compliance_results) - non_compliant_count) / 
                    len(compliance_results) * 100
                )
            },
            'detailed_results': compliance_results,
            'recommendation': self.get_recommendation(compliance_results)
        }
        
        return report
    
    def get_recommendation(self, results: List[Dict]) -> str:
        """Generate overall recommendation based on compliance results"""
        non_compliant_count = sum(
            1 for result in results 
            if result['compliance_status'] == 'Non-Compliant'
        )
        
        if non_compliant_count == 0:
            return "Prospectus appears to be fully compliant with ICDR 2018 regulations"
        elif non_compliant_count <= 2:
            return "Minor modifications needed for full ICDR compliance"
        else:
            return "Significant revisions needed to meet ICDR requirements"

def main():
    """Main execution function"""
    try:
        api_token = "nugen-CnStpNdbBczk3d8SZMhmnw"
        icdr_path = 'ICDR.pdf'
        prospectus_path = 'NTPC Green Energy Ltd - Prospectus.pdf'
        
        checker = ICDRComplianceChecker(icdr_path, api_token)
        report = checker.check_prospectus_compliance(prospectus_path)
        
        # Save report
        output_path = f'icdr_compliance_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"ICDR Compliance report generated: {output_path}")
        
    except Exception as e:
        print(f"Error in compliance checking: {str(e)}")

if __name__ == "__main__":
    main()