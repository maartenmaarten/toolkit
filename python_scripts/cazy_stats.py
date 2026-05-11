"""
CAZy Data Analysis and Visualization Script

This script analyzes large CAZy (Carbohydrate-Active Enzymes) TSV files
and generates comprehensive statistics and visualizations including:
- EC number distributions
- Taxonomic diversity analysis
- CAZy family distributions
- Domain composition
- Activity analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


class CAZyAnalyzer:
    """Comprehensive analyzer for CAZy enzyme data."""
    
    def __init__(self, filepath: str):
        """
        Initialize the analyzer with a CAZy TSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to the CAZy TSV file
        """
        self.filepath = Path(filepath)
        self.df = None
        self.output_dir = self.filepath.parent / "cazy_analysis_results"
        self.output_dir.mkdir(exist_ok=True)
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the CAZy data."""
        print(f"Loading data from {self.filepath}...")
        self.df = pd.read_csv(self.filepath, sep='\t', low_memory=False)
        print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
        print(f"\nColumn names: {list(self.df.columns)}")
    
    def get_basic_stats(self) -> Dict:
        """Calculate basic statistics about the dataset."""
        stats = {
            'total_entries': len(self.df),
            'unique_cazy_families': self.df['cazy_family'].nunique(),
            'unique_organisms': self.df['organism'].nunique(),
            'unique_domains': self.df['d__domain'].nunique(),
            'unique_taxa': {
                'domain': self.df['d__domain'].nunique(),
                'phylum': self.df['p__phylum'].nunique(),
                'class': self.df['c__class'].nunique(),
                'order': self.df['o__order'].nunique(),
                'family': self.df['f__family'].nunique(),
                'genus': self.df['g__genus'].nunique(),
                'species': self.df['s__species'].nunique(),
            }
        }
        return stats
    
    def print_basic_stats(self):
        """Print basic statistics to console."""
        stats = self.get_basic_stats()
        print("\n" + "="*60)
        print("BASIC STATISTICS")
        print("="*60)
        print(f"Total entries: {stats['total_entries']:,}")
        print(f"Unique CAZy families: {stats['unique_cazy_families']}")
        print(f"Unique organisms: {stats['unique_organisms']}")
        print(f"Unique domains: {stats['unique_domains']}")
        print("\nTaxonomic diversity:")
        for tax_level, count in stats['unique_taxa'].items():
            print(f"  - {tax_level}: {count}")
    
    def analyze_ec_distribution(self) -> Tuple[pd.Series, int]:
        """
        Analyze EC number distribution.
        
        Returns:
        --------
        pd.Series : EC number value counts
        int : Number of entries with EC numbers
        """
        # Remove NaN and empty strings
        ec_data = self.df['EC'].dropna()
        ec_data = ec_data[ec_data.str.strip() != '']
        
        # Handle multiple EC numbers (comma or semicolon separated)
        all_ecs = []
        for ec_str in ec_data:
            ecs = [e.strip() for e in str(ec_str).replace(';', ',').split(',')]
            all_ecs.extend(ecs)
        
        ec_counts = pd.Series(all_ecs).value_counts()
        return ec_counts, len(ec_data)
    
    def analyze_taxonomy_distribution(self) -> Dict:
        """Analyze distribution across taxonomic levels."""
        tax_dist = {
            'domain': self.df['d__domain'].value_counts(),
            'phylum': self.df['p__phylum'].value_counts(),
            'class': self.df['c__class'].value_counts(),
            'order': self.df['o__order'].value_counts(),
            'family': self.df['f__family'].value_counts(),
        }
        return tax_dist
    
    def analyze_cazy_families(self) -> pd.Series:
        """Analyze CAZy family distribution."""
        return self.df['cazy_family'].value_counts()
    
    def analyze_domain_distribution(self) -> pd.Series:
        """Analyze domain (Bacteria/Archaea/Eukaryota) distribution from 'domain' column."""
        return self.df['domain'].value_counts()
    
    def analyze_activity_distribution(self) -> pd.Series:
        """Analyze activity name distribution from function annotations."""
        activity_data = self.df['Activity Name'].dropna()
        activity_data = activity_data[activity_data.astype(str).str.strip() != '']
        return activity_data.value_counts()
    
    def analyze_ec_coverage_by_family(self) -> pd.DataFrame:
        """
        Calculate EC number coverage percentage for each CAZy family.
        
        Returns:
        --------
        pd.DataFrame with columns: family, total_sequences, with_ec, coverage_percent
        """
        # Group by CAZy family
        family_stats = []
        
        for family in self.df['cazy_family'].unique():
            family_data = self.df[self.df['cazy_family'] == family]
            total = len(family_data)
            with_ec = family_data['EC'].notna().sum()
            coverage = (with_ec / total * 100) if total > 0 else 0
            
            family_stats.append({
                'family': family,
                'total_sequences': total,
                'with_ec': with_ec,
                'coverage_percent': coverage
            })
        
        result_df = pd.DataFrame(family_stats).sort_values('coverage_percent', ascending=False)
        return result_df
    
    def get_missing_data_report(self) -> pd.DataFrame:
        """Generate a report on missing data."""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        report = pd.DataFrame({
            'missing_count': missing,
            'missing_percentage': missing_pct
        }).sort_values('missing_count', ascending=False)
        return report[report['missing_count'] > 0]
    
    # ==================== VISUALIZATION METHODS ====================
    
    def plot_ec_distribution(self, top_n: int = 30):
        """
        Plot EC numbers distribution.
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top EC numbers to plot. Default is 30.
        """
        ec_counts, total_with_ec = self.analyze_ec_distribution()
        
        # Determine which EC numbers to plot
        ecs_to_plot = ec_counts.head(top_n)
        title_suffix = f'Top {top_n} EC Numbers'
        
        # Adjust figure height based on number of EC numbers
        num_ecs = len(ecs_to_plot)
        fig_height = max(8, min(num_ecs * 0.3, 50))  # Scale with number of ECs, max 50
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height))
        
        # Plot EC numbers using seaborn
        sns.barplot(x=ecs_to_plot.values, y=ecs_to_plot.index, ax=ax1, palette='Blues_r')
        ax1.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax1.set_ylabel('')
        ax1.set_title(f'{title_suffix} (n={total_with_ec})', fontsize=12, fontweight='bold')
        
        # EC number availability
        has_ec = len(self.analyze_ec_distribution()[0])
        no_ec = len(self.df) - has_ec
        ax2.pie([has_ec, no_ec], labels=['With EC', 'Without EC'], 
                autopct='%1.3f%%', colors=['#2ecc71', '#e74c3c'])
        ax2.set_title('EC Number Availability', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ec_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: ec_distribution.png")
        plt.close()
    
    def plot_cazy_families(self, top_n: int = 30):
        """Plot CAZy family distribution."""
        family_counts = self.analyze_cazy_families()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Top families using seaborn
        top_families = family_counts.head(top_n)
        sns.barplot(x=top_families.values, y=top_families.index, ax=ax1, palette='Spectral')
        ax1.set_xlabel('Count')
        ax1.set_ylabel('')
        ax1.set_title(f'Top {top_n} CAZy Families (Total: {len(family_counts)})', fontsize=12, fontweight='bold')
        
        # Distribution as pie (top 15)
        other_sum = family_counts.iloc[15:].sum()
        plot_data = pd.concat([family_counts.head(15), pd.Series({'Other': other_sum})])
        ax2.pie(plot_data, labels=plot_data.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('CAZy Family Distribution (Top 15 + Other)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cazy_families.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: cazy_families.png")
        plt.close()
    
    def plot_domain_distribution(self):
        """Plot domain distribution."""
        domain_counts = self.analyze_domain_distribution()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar plot using seaborn
        sns.barplot(x=domain_counts.index, y=domain_counts.values, ax=ax1, palette='Set2')
        ax1.set_xlabel('Domain', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution by Domain', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Pie chart
        ax2.pie(domain_counts, labels=domain_counts.index, autopct='%1.1f%%')
        ax2.set_title('Domain Composition', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: domain_distribution.png")
        plt.close()
    
    def plot_taxonomy_distribution(self, tax_level: str = 'phylum', top_n: int = 15):
        """
        Plot taxonomic distribution at specified level.
        
        Parameters:
        -----------
        tax_level : str
            Taxonomic level: 'domain', 'phylum', 'class', 'order', 'family'
        top_n : int
            Number of top taxa to display
        """
        tax_dist = self.analyze_taxonomy_distribution()
        
        if tax_level not in tax_dist:
            print(f"Warning: {tax_level} not found in data")
            return
        
        top_taxa = tax_dist[tax_level].head(top_n)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(x=top_taxa.values, y=top_taxa.index, ax=ax, palette='RdYlGn_r')
        ax.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax.set_ylabel('')
        ax.set_title(f'Top {top_n} {tax_level.capitalize()} (Total: {tax_dist[tax_level].nunique()})', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'taxonomy_{tax_level}.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: taxonomy_{tax_level}.png")
        plt.close()
    
    def plot_all_taxonomy_levels(self, top_n: int = 15):
        """Plot all taxonomic levels."""
        for tax_level in ['domain', 'phylum', 'class', 'order', 'family']:
            self.plot_taxonomy_distribution(tax_level=tax_level, top_n=top_n)
    
    def plot_cazy_by_domain(self):
        """Plot CAZy families broken down by domain."""
        cazy_domain = pd.crosstab(self.df['cazy_family'], self.df['domain'])
        top_families = self.df['cazy_family'].value_counts().head(20).index
        
        cazy_domain_top = cazy_domain.loc[top_families].iloc[::-1]  # Reverse order so most common is at top
        
        fig, ax = plt.subplots(figsize=(12, 10))
        cazy_domain_top.plot(kind='barh', stacked=True, ax=ax, 
                             color=sns.color_palette('husl', len(cazy_domain_top.columns)))
        ax.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax.set_ylabel('')
        ax.set_title('Top 20 CAZy Families by Domain', fontsize=12, fontweight='bold')
        ax.legend(title='Domain', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cazy_by_domain.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: cazy_by_domain.png")
        plt.close()
    
    def plot_ec_coverage_distribution(self):
        """Plot boxplot of EC coverage distribution across all CAZy families."""
        ec_coverage = self.analyze_ec_coverage_by_family()
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Boxplot of coverage distribution
        bp = axes[0].boxplot(ec_coverage['coverage_percent'], vert=False, 
                            patch_artist=True, widths=0.5)
        
        # Color the boxplot
        for patch in bp['boxes']:
            patch.set_facecolor('#FF6B6B')
            patch.set_alpha(0.7)
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=2)
        
        axes[0].set_xlabel('EC Coverage (%)', fontsize=11, fontweight='bold')
        axes[0].set_title('Distribution of EC Annotation Coverage Across All CAZy Families', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlim(0, 105)
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].set_yticks([])
        
        # Add statistics text
        stats_text = f"""
        Total families: {len(ec_coverage)}
        Mean coverage: {ec_coverage['coverage_percent'].mean():.1f}%
        Median coverage: {ec_coverage['coverage_percent'].median():.1f}%
        Min coverage: {ec_coverage['coverage_percent'].min():.1f}%
        Max coverage: {ec_coverage['coverage_percent'].max():.1f}%
        Std dev: {ec_coverage['coverage_percent'].std():.1f}%
        """
        axes[0].text(0.98, 0.98, stats_text, transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    family='monospace')
        
        # Plot 2: Histogram of coverage distribution
        axes[1].hist(ec_coverage['coverage_percent'], bins=30, edgecolor='black', 
                    color='#FF6B6B', alpha=0.7)
        axes[1].set_xlabel('EC Coverage (%)', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Number of Families', fontsize=11, fontweight='bold')
        axes[1].set_title('Histogram of EC Coverage Distribution', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'ec_coverage_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: ec_coverage_distribution.png")
    
    def plot_organisms_per_family(self, top_n: int = 20):
        """Plot organism diversity per CAZy family."""
        organisms_per_family = self.df.groupby('cazy_family')['organism'].nunique().sort_values(ascending=False)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        top_orgs = organisms_per_family.head(top_n)
        sns.barplot(x=top_orgs.values, y=top_orgs.index, ax=ax, palette='viridis')
        ax.set_xlabel('Number of Unique Organisms', fontsize=11, fontweight='bold')
        ax.set_ylabel('')
        ax.set_title(f'Top {top_n} CAZy Families by Organism Diversity', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'organisms_per_family.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: organisms_per_family.png")
        plt.close()
    
    def plot_missing_data(self):
        """Plot missing data report."""
        missing_report = self.get_missing_data_report()
        
        if len(missing_report) == 0:
            print("No missing data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        missing_sorted = missing_report['missing_percentage'].sort_values()
        sns.barplot(x=missing_sorted.values, y=missing_sorted.index, ax=ax, palette='YlOrRd')
        ax.set_xlabel('Missing Data (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('')
        ax.set_title('Data Completeness by Column', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'missing_data.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: missing_data.png")
        plt.close()
    
    def plot_activity_distribution(self, top_n: int = 20):
        """Plot activity name distribution from function annotations."""
        activity_counts = self.analyze_activity_distribution()
        
        if len(activity_counts) == 0:
            print("No activity data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(14, 8))
        top_activities = activity_counts.head(top_n)
        sns.barplot(x=top_activities.values, y=top_activities.index, ax=ax, palette='coolwarm')
        ax.set_xlabel('Count', fontsize=11, fontweight='bold')
        ax.set_ylabel('')
        ax.set_title(f'Top {top_n} Enzyme Activities', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'activity_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: activity_distribution.png")
        plt.close()
    
    # ==================== REPORT GENERATION ====================
    
    def generate_csv_reports(self):
        """Generate CSV reports of key analyses."""
        # EC distribution
        ec_counts, _ = self.analyze_ec_distribution()
        ec_counts.to_csv(self.output_dir / 'ec_distribution.csv', header=['count'])
        print(f"✓ Saved: ec_distribution.csv")
        
        # EC coverage by family
        ec_coverage = self.analyze_ec_coverage_by_family()
        ec_coverage.to_csv(self.output_dir / 'ec_coverage_by_family.csv', index=False)
        print(f"✓ Saved: ec_coverage_by_family.csv")
        
        # CAZy families
        family_counts = self.analyze_cazy_families()
        family_counts.to_csv(self.output_dir / 'cazy_families.csv', header=['count'])
        print(f"✓ Saved: cazy_families.csv")
        
        # Domain distribution
        domain_counts = self.analyze_domain_distribution()
        domain_counts.to_csv(self.output_dir / 'domain_distribution.csv', header=['count'])
        print(f"✓ Saved: domain_distribution.csv")
        
        # Activity distribution
        activity_counts = self.analyze_activity_distribution()
        activity_counts.to_csv(self.output_dir / 'activity_distribution.csv', header=['count'])
        print(f"✓ Saved: activity_distribution.csv")
        
        # Taxonomy distributions
        tax_dist = self.analyze_taxonomy_distribution()
        for tax_level, counts in tax_dist.items():
            counts.to_csv(self.output_dir / f'taxonomy_{tax_level}.csv', header=['count'])
        print(f"✓ Saved: taxonomy_*.csv files")
        
        # Missing data report
        missing_report = self.get_missing_data_report()
        missing_report.to_csv(self.output_dir / 'missing_data_report.csv')
        print(f"✓ Saved: missing_data_report.csv")
    
    def generate_text_report(self):
        """Generate a comprehensive text report."""
        report_path = self.output_dir / 'analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("CAZy DATABASE ANALYSIS REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Basic stats
            stats = self.get_basic_stats()
            f.write("BASIC STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Total entries: {stats['total_entries']:,}\n")
            f.write(f"Unique CAZy families: {stats['unique_cazy_families']}\n")
            f.write(f"Unique organisms: {stats['unique_organisms']}\n")
            f.write(f"Unique domains: {stats['unique_domains']}\n")
            f.write("\nTaxonomic diversity:\n")
            for tax_level, count in stats['unique_taxa'].items():
                f.write(f"  - {tax_level}: {count}\n")
            
            # EC numbers
            f.write("\n" + "="*70 + "\n")
            f.write("EC NUMBER ANALYSIS\n")
            f.write("-"*70 + "\n")
            ec_counts, total_with_ec = self.analyze_ec_distribution()
            f.write(f"Entries with EC numbers: {total_with_ec} ({total_with_ec/len(self.df)*100:.1f}%)\n")
            f.write(f"Unique EC numbers: {len(ec_counts)}\n")
            f.write("\nTop 20 EC numbers:\n")
            for i, (ec, count) in enumerate(ec_counts.head(20).items(), 1):
                f.write(f"  {i:2d}. {ec}: {count}\n")
            
            # CAZy families
            f.write("\n" + "="*70 + "\n")
            f.write("CAZy FAMILY ANALYSIS\n")
            f.write("-"*70 + "\n")
            family_counts = self.analyze_cazy_families()
            f.write(f"Total unique families: {len(family_counts)}\n")
            f.write("\nTop 30 families:\n")
            for i, (family, count) in enumerate(family_counts.head(30).items(), 1):
                f.write(f"  {i:2d}. {family}: {count}\n")
            
            # Domain
            f.write("\n" + "="*70 + "\n")
            f.write("DOMAIN (BACTERIA/ARCHAEA/EUKARYOTA) ANALYSIS\n")
            f.write("-"*70 + "\n")
            domain_counts = self.analyze_domain_distribution()
            for domain, count in domain_counts.items():
                pct = count / len(self.df) * 100
                f.write(f"  {domain}: {count} ({pct:.1f}%)\n")
            
            # Activity analysis
            f.write("\n" + "="*70 + "\n")
            f.write("ACTIVITY ANALYSIS\n")
            f.write("-"*70 + "\n")
            activity_counts = self.analyze_activity_distribution()
            f.write(f"Entries with activity annotation: {len(activity_counts)}\n")
            f.write("\nTop 20 activities:\n")
            for i, (activity, count) in enumerate(activity_counts.head(20).items(), 1):
                f.write(f"  {i:2d}. {activity}: {count}\n")
            
            # EC coverage by family
            f.write("\n" + "="*70 + "\n")
            f.write("EC ANNOTATION COVERAGE BY CAZYME FAMILY\n")
            f.write("-"*70 + "\n")
            ec_coverage = self.analyze_ec_coverage_by_family()
            f.write("\nTop 20 families by EC coverage:\n")
            for i, row in ec_coverage.head(20).iterrows():
                f.write(f"  {row['family']}: {row['coverage_percent']:.1f}% ({row['with_ec']}/{row['total_sequences']})\n")
            
            # Missing data
            f.write("\n" + "="*70 + "\n")
            f.write("DATA QUALITY\n")
            f.write("-"*70 + "\n")
            missing_report = self.get_missing_data_report()
            if len(missing_report) > 0:
                f.write("Columns with missing data:\n")
                for col, row in missing_report.iterrows():
                    f.write(f"  {col}: {row['missing_count']:,} ({row['missing_percentage']:.1f}%)\n")
            else:
                f.write("No missing data found.\n")
        
        print(f"✓ Saved: analysis_report.txt")
        return report_path
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("\n" + "="*60)
        print("STARTING FULL ANALYSIS")
        print("="*60)
        
        # Print stats
        self.print_basic_stats()
        
        # Generate reports
        print("\nGenerating reports...")
        self.generate_text_report()
        self.generate_csv_reports()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_ec_distribution()
        self.plot_cazy_families()
        self.plot_domain_distribution()
        self.plot_all_taxonomy_levels()
        self.plot_ec_coverage_distribution()
        self.plot_organisms_per_family()
        self.plot_missing_data()
        self.plot_activity_distribution()
        
        print("\n" + "="*60)
        print(f"Analysis complete! Results saved to:")
        print(f"  {self.output_dir}")
        print("="*60)


def main():
    """Main function to run the analysis."""
    # Specify your input file path here
    input_file = "/Users/maartenboneschansker/Documents/cazy_data/CAZy_cazymes/cazy/cazy_data_20251215.tax.fun.tsv"
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        return
    
    analyzer = CAZyAnalyzer(input_file)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
