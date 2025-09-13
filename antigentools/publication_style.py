#!/usr/bin/env python3
"""
Publication-ready figure styling for scientific publications.

This module provides a comprehensive set of tools for creating publication-quality
figures with consistent styling, proper formatting, and journal-specific requirements.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.figure import Figure
from matplotlib.axes import Axes


# Color palettes
COLORBLIND_FRIENDLY = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#CC78BC',
    'purple': '#CA9161',
    'brown': '#949494',
    'pink': '#FBAFE4',
    'gray': '#949494',
    'yellow': '#ECE133',
    'light_blue': '#56B4E9'
}

NATURE_COLORS = {
    'primary': '#000000',
    'secondary': '#666666',
    'accent1': '#E64B35',
    'accent2': '#4DBBD5',
    'accent3': '#00A087',
    'accent4': '#3C5488',
    'accent5': '#F39B7F',
    'accent6': '#8491B4',
    'accent7': '#91D1C2',
    'accent8': '#DC0000'
}

SCIENCE_COLORS = {
    'primary': '#000000',
    'secondary': '#666666',
    'blue': '#0055A4',
    'red': '#EF3340',
    'yellow': '#F2B705',
    'green': '#00A652',
    'purple': '#7B3294',
    'orange': '#FF6600'
}

# Journal style presets
JOURNAL_STYLES = {
    'nature': {
        'font_family': 'Arial',
        'font_size_base': 7,
        'font_size_labels': 8,
        'font_size_title': 9,
        'font_size_panel': 10,
        'line_width': 0.5,
        'tick_width': 0.5,
        'spine_width': 0.5,
        'figure_width_single': 3.35,  # inches (85mm)
        'figure_width_double': 7.08,  # inches (180mm)
        'dpi': 300,
        'colors': NATURE_COLORS
    },
    'science': {
        'font_family': 'Arial',
        'font_size_base': 8,
        'font_size_labels': 9,
        'font_size_title': 10,
        'font_size_panel': 11,
        'line_width': 0.75,
        'tick_width': 0.5,
        'spine_width': 0.75,
        'figure_width_single': 3.42,  # inches (87mm)
        'figure_width_double': 7.08,  # inches (180mm)
        'dpi': 300,
        'colors': SCIENCE_COLORS
    },
    'plos': {
        'font_family': 'Arial',
        'font_size_base': 8,
        'font_size_labels': 10,
        'font_size_title': 12,
        'font_size_panel': 12,
        'line_width': 1.0,
        'tick_width': 0.75,
        'spine_width': 1.0,
        'figure_width_single': 3.27,  # inches (83mm)
        'figure_width_double': 6.83,  # inches (173.5mm)
        'dpi': 300,
        'colors': COLORBLIND_FRIENDLY
    },
    'cell': {
        'font_family': 'Arial',
        'font_size_base': 6,
        'font_size_labels': 7,
        'font_size_title': 8,
        'font_size_panel': 9,
        'line_width': 0.5,
        'tick_width': 0.5,
        'spine_width': 0.5,
        'figure_width_single': 3.35,  # inches (85mm)
        'figure_width_double': 7.00,  # inches (178mm)
        'dpi': 300,
        'colors': NATURE_COLORS
    },
    'elife': {
        'font_family': 'Arial',
        'font_size_base': 9,
        'font_size_labels': 10,
        'font_size_title': 11,
        'font_size_panel': 12,
        'line_width': 1.0,
        'tick_width': 0.75,
        'spine_width': 1.0,
        'figure_width_single': 3.5,
        'figure_width_double': 7.0,
        'dpi': 300,
        'colors': COLORBLIND_FRIENDLY
    },
    'biorxiv': {
        'font_family': 'Arial',
        'font_size_base': 10,       # Larger for online readability
        'font_size_labels': 12,     # More prominent labels
        'font_size_title': 14,      # Titles allowed and encouraged
        'font_size_panel': 14,      # Bold panel labels
        'line_width': 1.2,          # Slightly thicker lines for screen viewing
        'tick_width': 0.8,
        'spine_width': 1.0,
        'figure_width_single': 4.0, # Wider for better online viewing
        'figure_width_double': 8.0, # More generous width
        'dpi': 300,                 # Good balance of quality and file size
        'colors': COLORBLIND_FRIENDLY  # Accessibility important for broad audience
    }
}


class PublicationFigureStyler:
    """
    A class for creating publication-ready figures with consistent styling.
    
    This class provides methods for setting up figures according to journal
    specifications, applying consistent styling, and adding common elements
    like panel labels and scale bars.
    
    Attributes
    ----------
    journal : str
        The target journal style ('nature', 'science', 'plos', etc.)
    style : dict
        Style parameters for the specified journal
    """
    
    def __init__(self, journal: str = 'nature'):
        """
        Initialize the figure styler with a specific journal style.
        
        Parameters
        ----------
        journal : str, default='nature'
            The target journal style. Options: 'nature', 'science', 'plos', 'cell', 'elife'
        """
        if journal not in JOURNAL_STYLES:
            raise ValueError(f"Unknown journal: {journal}. Available options: {list(JOURNAL_STYLES.keys())}")
        
        self.journal = journal
        self.style = JOURNAL_STYLES[journal].copy()
        self._original_rcParams = {}
        
    def setup_figure(self, 
                    n_rows: int = 1, 
                    n_cols: int = 1,
                    width: Optional[float] = None,
                    height_ratios: Optional[List[float]] = None,
                    width_ratios: Optional[List[float]] = None,
                    panel_spacing: float = 0.4) -> Tuple[Figure, Union[Axes, np.ndarray]]:
        """
        Create a new figure with the appropriate size and layout.
        
        Parameters
        ----------
        n_rows : int, default=1
            Number of subplot rows
        n_cols : int, default=1
            Number of subplot columns
        width : float, optional
            Figure width in inches. If None, uses journal default
        height_ratios : list of float, optional
            Height ratios for subplot rows
        width_ratios : list of float, optional
            Width ratios for subplot columns
        panel_spacing : float, default=0.4
            Spacing between panels in inches
        
        Returns
        -------
        fig : Figure
            The matplotlib figure object
        axes : Axes or array of Axes
            The axes object(s)
        """
        # Determine figure width
        if width is None:
            width = self.style['figure_width_double'] if n_cols > 1 else self.style['figure_width_single']
        
        # Calculate figure height based on golden ratio or custom ratios
        if height_ratios:
            total_height_ratio = sum(height_ratios)
            height = width * total_height_ratio / n_cols * 0.8
        else:
            height = width * n_rows / n_cols * 0.618  # Golden ratio
        
        # Create figure with GridSpec for better control
        fig = plt.figure(figsize=(width, height), dpi=self.style['dpi'])
        
        gs = GridSpec(n_rows, n_cols, figure=fig,
                      height_ratios=height_ratios,
                      width_ratios=width_ratios,
                      hspace=panel_spacing/height,
                      wspace=panel_spacing/width)
        
        # Create axes
        if n_rows == 1 and n_cols == 1:
            axes = fig.add_subplot(gs[0, 0])
        else:
            axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(n_cols)] 
                            for i in range(n_rows)])
            if n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
        
        return fig, axes
    
    def apply_style(self, ax: Axes, 
                   spine_position: str = 'left_bottom',
                   grid: bool = False,
                   minor_ticks: bool = False) -> None:
        """
        Apply journal-specific styling to an axes object.
        
        Parameters
        ----------
        ax : Axes
            The matplotlib axes to style
        spine_position : str, default='left_bottom'
            Which spines to show. Options: 'all', 'left_bottom', 'none', 'bottom'
        grid : bool, default=False
            Whether to show grid lines
        minor_ticks : bool, default=False
            Whether to show minor tick marks
        """
        # Set spine visibility and width
        spines_to_show = {
            'all': ['left', 'right', 'top', 'bottom'],
            'left_bottom': ['left', 'bottom'],
            'bottom': ['bottom'],
            'none': []
        }
        
        for spine in ['left', 'right', 'top', 'bottom']:
            if spine in spines_to_show.get(spine_position, ['left', 'bottom']):
                ax.spines[spine].set_linewidth(self.style['spine_width'])
                ax.spines[spine].set_visible(True)
            else:
                ax.spines[spine].set_visible(False)
        
        # Set tick parameters
        ax.tick_params(axis='both', 
                      width=self.style['tick_width'],
                      labelsize=self.style['font_size_base'],
                      length=3,
                      pad=2)
        
        if minor_ticks:
            ax.minorticks_on()
            ax.tick_params(axis='both', which='minor', 
                          width=self.style['tick_width'] * 0.75,
                          length=2)
        
        # Set grid
        if grid:
            ax.grid(True, linewidth=0.25, alpha=0.5, linestyle='-', color='gray')
            ax.set_axisbelow(True)
        
        # Set label font sizes
        ax.xaxis.label.set_fontsize(self.style['font_size_labels'])
        ax.yaxis.label.set_fontsize(self.style['font_size_labels'])
        
    def add_panel_label(self, ax: Axes, label: str, 
                       location: str = 'top_left',
                       offset: Tuple[float, float] = (-0.15, 1.05)) -> None:
        """
        Add a panel label (A, B, C, etc.) to an axes.
        
        Parameters
        ----------
        ax : Axes
            The axes to label
        label : str
            The panel label (e.g., 'A', 'B', 'C')
        location : str, default='top_left'
            Label position. Options: 'top_left', 'top_right', 'custom'
        offset : tuple of float, default=(-0.15, 1.05)
            Custom offset for label position (x, y) in axes fraction
        """
        locations = {
            'top_left': (-0.15, 1.05),
            'top_right': (1.05, 1.05),
            'custom': offset
        }
        
        x, y = locations.get(location, offset)
        
        ax.text(x, y, label, 
                transform=ax.transAxes,
                fontsize=self.style['font_size_panel'],
                fontweight='bold',
                va='bottom',
                ha='left' if x < 0.5 else 'right')
    
    def format_axis(self, ax: Axes, 
                   x_label: Optional[str] = None,
                   y_label: Optional[str] = None,
                   x_lim: Optional[Tuple[float, float]] = None,
                   y_lim: Optional[Tuple[float, float]] = None,
                   x_ticks: Optional[List[float]] = None,
                   y_ticks: Optional[List[float]] = None,
                   x_tick_labels: Optional[List[str]] = None,
                   y_tick_labels: Optional[List[str]] = None,
                   rotate_x_labels: float = 0,
                   sci_notation: bool = False,
                   log_scale_x: bool = False,
                   log_scale_y: bool = False) -> None:
        """
        Format axis with labels, limits, and ticks.
        
        Parameters
        ----------
        ax : Axes
            The axes to format
        x_label, y_label : str, optional
            Axis labels
        x_lim, y_lim : tuple of float, optional
            Axis limits
        x_ticks, y_ticks : list of float, optional
            Tick positions
        x_tick_labels, y_tick_labels : list of str, optional
            Tick labels
        rotate_x_labels : float, default=0
            Rotation angle for x-axis labels
        sci_notation : bool, default=False
            Whether to use scientific notation
        log_scale_x, log_scale_y : bool, default=False
            Whether to use log scale
        """
        # Set labels
        if x_label:
            ax.set_xlabel(x_label, fontsize=self.style['font_size_labels'])
        if y_label:
            ax.set_ylabel(y_label, fontsize=self.style['font_size_labels'])
        
        # Set limits
        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)
        
        # Set scale
        if log_scale_x:
            ax.set_xscale('log')
        if log_scale_y:
            ax.set_yscale('log')
        
        # Set ticks
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        
        # Set tick labels
        if x_tick_labels is not None:
            ax.set_xticklabels(x_tick_labels)
        if y_tick_labels is not None:
            ax.set_yticklabels(y_tick_labels)
        
        # Rotate x labels if needed
        if rotate_x_labels:
            ax.tick_params(axis='x', rotation=rotate_x_labels)
        
        # Scientific notation
        if sci_notation:
            ax.ticklabel_format(style='sci', scilimits=(-2, 2))
    
    def add_legend(self, ax: Axes,
                  handles: Optional[List] = None,
                  labels: Optional[List[str]] = None,
                  location: str = 'best',
                  n_cols: int = 1,
                  frameon: bool = False,
                  title: Optional[str] = None,
                  bbox_to_anchor: Optional[Tuple[float, float]] = None) -> None:
        """
        Add a formatted legend to the axes.
        
        Parameters
        ----------
        ax : Axes
            The axes to add legend to
        handles : list, optional
            Legend handles. If None, uses ax.get_legend_handles_labels()
        labels : list of str, optional
            Legend labels
        location : str, default='best'
            Legend location
        n_cols : int, default=1
            Number of columns in legend
        frameon : bool, default=False
            Whether to show legend frame
        title : str, optional
            Legend title
        bbox_to_anchor : tuple of float, optional
            Custom legend position
        """
        if handles is None and labels is None:
            handles, labels = ax.get_legend_handles_labels()
        
        legend_params = {
            'fontsize': self.style['font_size_base'],
            'frameon': frameon,
            'ncol': n_cols,
            'loc': location,
            'borderaxespad': 0.5,
            'columnspacing': 1.0,
            'handlelength': 1.5,
            'handleheight': 0.7,
            'handletextpad': 0.5
        }
        
        if title:
            legend_params['title'] = title
            legend_params['title_fontsize'] = self.style['font_size_labels']
        
        if bbox_to_anchor:
            legend_params['bbox_to_anchor'] = bbox_to_anchor
        
        if handles and labels:
            ax.legend(handles, labels, **legend_params)
        else:
            ax.legend(**legend_params)
    
    def add_scale_bar(self, ax: Axes,
                     length: float,
                     label: str,
                     location: str = 'lower right',
                     orientation: str = 'horizontal',
                     color: str = 'black',
                     linewidth: Optional[float] = None) -> None:
        """
        Add a scale bar to the axes.
        
        Parameters
        ----------
        ax : Axes
            The axes to add scale bar to
        length : float
            Length of scale bar in data units
        label : str
            Scale bar label
        location : str, default='lower right'
            Scale bar location
        orientation : str, default='horizontal'
            'horizontal' or 'vertical'
        color : str, default='black'
            Scale bar color
        linewidth : float, optional
            Scale bar line width. If None, uses style default
        """
        if linewidth is None:
            linewidth = self.style['line_width'] * 2
        
        # Get axes limits
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Calculate positions based on location
        locations = {
            'lower right': (0.95, 0.05),
            'lower left': (0.05, 0.05),
            'upper right': (0.95, 0.95),
            'upper left': (0.05, 0.95)
        }
        
        x_frac, y_frac = locations.get(location, (0.95, 0.05))
        
        if orientation == 'horizontal':
            x_data = xlim[0] + x_frac * (xlim[1] - xlim[0])
            y_data = ylim[0] + y_frac * (ylim[1] - ylim[0])
            
            # Adjust x position to right-align if on right side
            if x_frac > 0.5:
                x_data -= length
            
            ax.plot([x_data, x_data + length], [y_data, y_data],
                   color=color, linewidth=linewidth, solid_capstyle='butt')
            
            # Add label
            ax.text(x_data + length/2, y_data - 0.02*(ylim[1] - ylim[0]),
                   label, ha='center', va='top',
                   fontsize=self.style['font_size_base'])
        
        else:  # vertical
            x_data = xlim[0] + x_frac * (xlim[1] - xlim[0])
            y_data = ylim[0] + y_frac * (ylim[1] - ylim[0])
            
            # Adjust y position to top-align if on upper side
            if y_frac > 0.5:
                y_data -= length
            
            ax.plot([x_data, x_data], [y_data, y_data + length],
                   color=color, linewidth=linewidth, solid_capstyle='butt')
            
            # Add label
            ax.text(x_data + 0.02*(xlim[1] - xlim[0]), y_data + length/2,
                   label, ha='left', va='center',
                   fontsize=self.style['font_size_base'], rotation=90)
    
    def save_figure(self, fig: Figure, 
                   filename: str,
                   formats: List[str] = ['pdf', 'png'],
                   dpi: Optional[int] = None,
                   bbox_inches: str = 'tight',
                   transparent: bool = False) -> None:
        """
        Save figure in publication-ready formats.
        
        Parameters
        ----------
        fig : Figure
            The figure to save
        filename : str
            Base filename (without extension)
        formats : list of str, default=['pdf', 'png']
            File formats to save
        dpi : int, optional
            DPI for raster formats. If None, uses style default
        bbox_inches : str, default='tight'
            Bounding box setting
        transparent : bool, default=False
            Whether to save with transparent background
        """
        if dpi is None:
            dpi = self.style['dpi']
        
        for fmt in formats:
            save_path = f"{filename}.{fmt}"
            fig.savefig(save_path,
                       format=fmt,
                       dpi=dpi,
                       bbox_inches=bbox_inches,
                       transparent=transparent,
                       facecolor='white' if not transparent else 'none')
            print(f"Saved: {save_path}")
    
    def get_colors(self, n: int = None, 
                  palette: str = 'default',
                  colorblind_safe: bool = True) -> List[str]:
        """
        Get a list of colors for plotting.
        
        Parameters
        ----------
        n : int, optional
            Number of colors needed. If None, returns all colors in palette
        palette : str, default='default'
            Color palette name. Options: 'default', 'sequential', 'diverging', 'qualitative'
        colorblind_safe : bool, default=True
            Whether to use colorblind-friendly colors
        
        Returns
        -------
        colors : list of str
            List of color codes
        """
        if palette == 'default':
            if colorblind_safe:
                colors = list(COLORBLIND_FRIENDLY.values())
            else:
                colors = list(self.style['colors'].values())
        elif palette == 'sequential':
            colors = plt.cm.viridis(np.linspace(0, 0.9, n or 10))
            colors = [mpl.colors.rgb2hex(c) for c in colors]
        elif palette == 'diverging':
            colors = plt.cm.RdBu_r(np.linspace(0, 1, n or 10))
            colors = [mpl.colors.rgb2hex(c) for c in colors]
        elif palette == 'qualitative':
            if colorblind_safe:
                colors = list(COLORBLIND_FRIENDLY.values())
            else:
                colors = plt.cm.tab10(np.arange(n or 10))
                colors = [mpl.colors.rgb2hex(c) for c in colors]
        
        if n and len(colors) < n:
            # Cycle through colors if we need more
            colors = colors * (n // len(colors) + 1)
        
        return colors[:n] if n else colors


@contextmanager
def publication_style(journal: str = 'nature', **kwargs):
    """
    Context manager for temporarily applying publication style settings.
    
    Parameters
    ----------
    journal : str, default='nature'
        The journal style to apply
    **kwargs : dict
        Additional matplotlib rcParams to set
    
    Examples
    --------
    >>> with publication_style('nature'):
    ...     fig, ax = plt.subplots()
    ...     ax.plot(x, y)
    ...     plt.savefig('figure.pdf')
    """
    # Store original rcParams
    original_params = plt.rcParams.copy()
    
    # Get journal style
    style = JOURNAL_STYLES.get(journal, JOURNAL_STYLES['nature'])
    
    try:
        # Apply style settings
        plt.rcParams['font.family'] = style['font_family']
        plt.rcParams['font.size'] = style['font_size_base']
        plt.rcParams['axes.linewidth'] = style['spine_width']
        plt.rcParams['axes.labelsize'] = style['font_size_labels']
        plt.rcParams['axes.titlesize'] = style['font_size_title']
        plt.rcParams['xtick.labelsize'] = style['font_size_base']
        plt.rcParams['ytick.labelsize'] = style['font_size_base']
        plt.rcParams['legend.fontsize'] = style['font_size_base']
        plt.rcParams['lines.linewidth'] = style['line_width']
        plt.rcParams['patch.linewidth'] = style['line_width']
        plt.rcParams['xtick.major.width'] = style['tick_width']
        plt.rcParams['ytick.major.width'] = style['tick_width']
        plt.rcParams['xtick.minor.width'] = style['tick_width'] * 0.75
        plt.rcParams['ytick.minor.width'] = style['tick_width'] * 0.75
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['xtick.minor.size'] = 2
        plt.rcParams['ytick.minor.size'] = 2
        
        # Apply any additional parameters
        for key, value in kwargs.items():
            plt.rcParams[key] = value
        
        yield
        
    finally:
        # Restore original settings
        plt.rcParams.update(original_params)


def create_multipanel_figure(n_panels: int,
                           panel_arrangement: Optional[str] = None,
                           journal: str = 'nature',
                           panel_labels: bool = True,
                           **fig_kwargs) -> Tuple[PublicationFigureStyler, Figure, Union[Axes, np.ndarray]]:
    """
    Convenience function to create a multi-panel figure with consistent styling.
    
    Parameters
    ----------
    n_panels : int
        Total number of panels
    panel_arrangement : str, optional
        Panel arrangement. Options: 'horizontal', 'vertical', 'square', or 'rows,cols' (e.g., '2,3')
    journal : str, default='nature'
        Target journal style
    panel_labels : bool, default=True
        Whether to add panel labels (A, B, C, etc.)
    **fig_kwargs : dict
        Additional arguments passed to setup_figure()
    
    Returns
    -------
    styler : PublicationFigureStyler
        The figure styler instance
    fig : Figure
        The matplotlib figure
    axes : Axes or array of Axes
        The axes objects
    
    Examples
    --------
    >>> styler, fig, axes = create_multipanel_figure(4, 'square', journal='nature')
    >>> for i, ax in enumerate(axes.flat):
    ...     ax.plot(x, y[i])
    ...     styler.apply_style(ax)
    """
    # Determine panel arrangement
    if panel_arrangement is None:
        # Auto-arrange in roughly square layout
        n_cols = int(np.ceil(np.sqrt(n_panels)))
        n_rows = int(np.ceil(n_panels / n_cols))
    elif panel_arrangement == 'horizontal':
        n_rows, n_cols = 1, n_panels
    elif panel_arrangement == 'vertical':
        n_rows, n_cols = n_panels, 1
    elif panel_arrangement == 'square':
        n_cols = int(np.ceil(np.sqrt(n_panels)))
        n_rows = int(np.ceil(n_panels / n_cols))
    elif ',' in panel_arrangement:
        n_rows, n_cols = map(int, panel_arrangement.split(','))
    else:
        raise ValueError(f"Unknown panel arrangement: {panel_arrangement}")
    
    # Create styler and figure
    styler = PublicationFigureStyler(journal)
    fig, axes = styler.setup_figure(n_rows, n_cols, **fig_kwargs)
    
    # Add panel labels if requested
    if panel_labels:
        labels = [chr(ord('A') + i) for i in range(n_panels)]
        
        if isinstance(axes, np.ndarray):
            for i, (ax, label) in enumerate(zip(axes.flat, labels)):
                if i < n_panels:
                    styler.add_panel_label(ax, label)
                else:
                    ax.set_visible(False)  # Hide extra panels
        else:
            styler.add_panel_label(axes, 'A')
    
    return styler, fig, axes


def format_significance_stars(p_value: float, 
                            thresholds: Dict[str, float] = None) -> str:
    """
    Convert p-value to significance stars.
    
    Parameters
    ----------
    p_value : float
        The p-value to convert
    thresholds : dict, optional
        Custom significance thresholds. Default: {0.001: '***', 0.01: '**', 0.05: '*'}
    
    Returns
    -------
    str
        Significance stars or 'ns' for non-significant
    """
    if thresholds is None:
        thresholds = {0.001: '***', 0.01: '**', 0.05: '*'}
    
    for threshold, stars in sorted(thresholds.items()):
        if p_value < threshold:
            return stars
    
    return 'ns'


def add_significance_bar(ax: Axes,
                        x1: float, x2: float, 
                        y: float,
                        p_value: float,
                        height: float = None,
                        text_offset: float = 0.02,
                        color: str = 'black',
                        linewidth: float = 1.0,
                        fontsize: int = None) -> None:
    """
    Add a significance bar with stars to indicate statistical significance.
    
    Parameters
    ----------
    ax : Axes
        The axes to add the significance bar to
    x1, x2 : float
        x-coordinates of the two points to compare
    y : float
        y-coordinate for the significance bar
    p_value : float
        The p-value for the comparison
    height : float, optional
        Height of the vertical lines at bar ends. If None, auto-calculated
    text_offset : float, default=0.02
        Vertical offset for significance text as fraction of y-axis range
    color : str, default='black'
        Color for the bar and text
    linewidth : float, default=1.0
        Line width for the bar
    fontsize : int, optional
        Font size for significance text
    """
    # Get significance stars
    sig_text = format_significance_stars(p_value)
    
    # Calculate height if not provided
    if height is None:
        ylim = ax.get_ylim()
        height = 0.01 * (ylim[1] - ylim[0])
    
    # Draw the bar
    ax.plot([x1, x1, x2, x2], [y - height, y, y, y - height],
            color=color, linewidth=linewidth)
    
    # Add significance text
    ylim = ax.get_ylim()
    y_text = y + text_offset * (ylim[1] - ylim[0])
    
    if fontsize is None:
        # Try to get fontsize from journal style if available
        fontsize = plt.rcParams.get('font.size', 8)
    
    ax.text((x1 + x2) / 2, y_text, sig_text,
            ha='center', va='bottom',
            color=color, fontsize=fontsize)


def apply_publication_styling_to_figure(fig: Figure, 
                                        journal: str = 'nature',
                                        panel_labels: bool = True,
                                        remove_titles: bool = True,
                                        legend_config: Optional[Dict] = None,
                                        spine_config: str = 'left_bottom',
                                        grid: bool = False) -> Figure:
    """
    Apply publication styling to an existing figure.
    
    This function takes an existing matplotlib figure and applies journal-specific
    styling without recreating the plots. It modifies fonts, line widths, colors,
    legends, and other aesthetic elements.
    
    Parameters
    ----------
    fig : Figure
        The matplotlib figure to style
    journal : str, default='nature'
        Target journal style ('nature', 'science', 'plos', 'cell', 'elife')
    panel_labels : bool, default=True
        Whether to add panel labels (A, B, C, etc.)
    remove_titles : bool, default=True
        Whether to remove subplot titles
    legend_config : dict, optional
        Legend configuration options:
        - 'position': bbox_to_anchor tuple or location string
        - 'ncol': number of columns
        - 'frameon': whether to show frame
        - 'remove_from_panels': list of panel indices to remove legends from
    spine_config : str, default='left_bottom'
        Spine configuration ('all', 'left_bottom', 'bottom', 'none')
    grid : bool, default=False
        Whether to show grid lines
        
    Returns
    -------
    Figure
        The styled figure (same object, modified in place)
        
    Examples
    --------
    >>> fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    >>> # ... plot data ...
    >>> styled_fig = apply_publication_styling_to_figure(fig, journal='plos')
    """
    # Get journal style
    if journal not in JOURNAL_STYLES:
        raise ValueError(f"Unknown journal: {journal}. Available: {list(JOURNAL_STYLES.keys())}")
    
    style = JOURNAL_STYLES[journal]
    
    # Apply style to all axes in the figure
    axes = fig.get_axes()
    
    for i, ax in enumerate(axes):
        # Apply basic styling
        _apply_axis_styling(ax, style, spine_config, grid)
        
        # Remove titles if requested
        if remove_titles:
            ax.set_title('')
        
        # Add panel labels if requested
        if panel_labels:
            label = chr(ord('A') + i)
            _add_panel_label(ax, label, style)
        
        # Configure legends
        if legend_config:
            _configure_legend(ax, legend_config, style, journal, i)
    
    # Apply figure-level styling
    _apply_figure_styling(fig, style)
    
    return fig


def _apply_axis_styling(ax: Axes, style: Dict, spine_config: str, grid: bool) -> None:
    """Apply styling to a single axes object."""
    # Set spine visibility and width
    spines_to_show = {
        'all': ['left', 'right', 'top', 'bottom'],
        'left_bottom': ['left', 'bottom'],
        'bottom': ['bottom'],
        'none': []
    }
    
    for spine in ['left', 'right', 'top', 'bottom']:
        if spine in spines_to_show.get(spine_config, ['left', 'bottom']):
            ax.spines[spine].set_linewidth(style['spine_width'])
            ax.spines[spine].set_visible(True)
        else:
            ax.spines[spine].set_visible(False)
    
    # Set tick parameters
    ax.tick_params(axis='both', 
                  width=style['tick_width'],
                  labelsize=style['font_size_base'],
                  length=3,
                  pad=2)
    
    # Set grid
    if grid:
        ax.grid(True, linewidth=0.25, alpha=0.5, linestyle='-', color='gray')
        ax.set_axisbelow(True)
    
    # Update label font sizes
    if ax.get_xlabel():
        ax.xaxis.label.set_fontsize(style['font_size_labels'])
    if ax.get_ylabel():
        ax.yaxis.label.set_fontsize(style['font_size_labels'])
    
    # Update line widths for existing plots
    for line in ax.get_lines():
        if line.get_linewidth() == plt.rcParams['lines.linewidth']:  # Only update if using default
            line.set_linewidth(style['line_width'])


def _add_panel_label(ax: Axes, label: str, style: Dict) -> None:
    """Add panel label to axes."""
    ax.text(-0.15, 1.05, label, 
            transform=ax.transAxes,
            fontsize=style['font_size_panel'],
            fontweight='bold',
            va='bottom',
            ha='left')


def _configure_legend(ax: Axes, legend_config: Dict, style: Dict, journal: str, panel_idx: int) -> None:
    """Configure legend for an axes."""
    # Remove legend from specified panels
    if 'remove_from_panels' in legend_config and panel_idx in legend_config['remove_from_panels']:
        legend = ax.get_legend()
        if legend:
            legend.remove()
        return
    
    # Get existing legend or create new one
    legend = ax.get_legend()
    if not legend:
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
    else:
        handles = legend.legendHandles
        labels = [t.get_text() for t in legend.get_texts()]
    
    # Configure legend parameters
    legend_params = {
        'fontsize': style['font_size_base'],
        'frameon': legend_config.get('frameon', False),
        'ncol': legend_config.get('ncol', 1),
        'borderaxespad': 0.5,
        'columnspacing': 1.0,
        'handlelength': 1.5,
        'handleheight': 0.7,
        'handletextpad': 0.5
    }
    
    # Set position
    if 'position' in legend_config:
        position = legend_config['position']
        if isinstance(position, tuple):
            legend_params['bbox_to_anchor'] = position
            legend_params['loc'] = 'upper center'
        else:
            legend_params['loc'] = position
    else:
        # Use journal-specific default positions
        journal_positions = {
            'nature': {'bbox_to_anchor': (0.5, -0.02), 'loc': 'upper center'},
            'science': {'bbox_to_anchor': (0.5, -0.02), 'loc': 'upper center'},
            'plos': {'bbox_to_anchor': (0.5, -0.025), 'loc': 'upper center'}
        }
        if journal in journal_positions:
            legend_params.update(journal_positions[journal])
    
    # Remove old legend and add new one
    if legend:
        legend.remove()
    
    new_legend = ax.legend(handles, labels, **legend_params)
    
    # Style the legend frame for PLOS
    if journal == 'plos' and legend_config.get('frameon', False):
        try:
            new_legend.get_frame().set_linewidth(0.5)
        except AttributeError:
            pass  # Skip if method doesn't exist


def _apply_figure_styling(fig: Figure, style: Dict) -> None:
    """Apply figure-level styling."""
    # Update all text elements in the figure
    for text in fig.findobj(plt.Text):
        if text.get_fontsize() == plt.rcParams['font.size']:  # Only update if using default
            text.set_fontsize(style['font_size_base'])
        # Set font family
        text.set_fontfamily(style['font_family'])


def style_figure_for_publication(fig: Figure, 
                                journal: str = 'nature',
                                remove_legend_from_panel_c: bool = True,
                                save_path: Optional[str] = None,
                                save_formats: List[str] = ['pdf', 'png'],
                                dpi: Optional[int] = None,
                                **kwargs) -> Figure:
    """
    Convenience function to style a multi-panel figure for publication.
    
    This is a wrapper around apply_publication_styling_to_figure with sensible
    defaults for the growth rate figures used in this project.
    
    Parameters
    ----------
    fig : Figure
        The matplotlib figure to style
    journal : str, default='nature'
        Target journal ('nature', 'science', 'plos')
    remove_legend_from_panel_c : bool, default=True
        Whether to remove legend from panel C/D (growth rate comparison)
    save_path : str, optional
        Path to save the figure (without extension). If None, figure is not saved.
    save_formats : list of str, default=['pdf', 'png']
        File formats to save. Each format will be saved with appropriate extension.
    dpi : int, optional
        DPI for saving. If None, uses journal-specific default.
    **kwargs : dict
        Additional arguments passed to apply_publication_styling_to_figure
        
    Returns
    -------
    Figure
        The styled figure
        
    Examples
    --------
    >>> # Create your plot normally
    >>> fig, axes = plt.subplots(1, 4, figsize=(12, 8))
    >>> plot_inferred_growth_rates(pivot_date, location, model, build, 
    ...                           growth_rates_df, color_map, fig=fig, axes=axes)
    >>> 
    >>> # Apply publication styling and save
    >>> styled_fig = style_figure_for_publication(
    ...     fig, 
    ...     journal='plos',
    ...     save_path='figures/growth_rates_plos',
    ...     save_formats=['pdf', 'png', 'svg']
    ... )
    >>> plt.show()
    """
    # Set up legend configuration - update for 4-panel layout
    legend_config = {
        'ncol': 2,
        'frameon': False,
    }
    
    if remove_legend_from_panel_c:
        # For 4-panel layout, panel D (index 3) is the growth rate comparison
        # For 3-panel layout, panel C (index 2) is the growth rate comparison
        num_panels = len(fig.get_axes())
        if num_panels >= 4:
            legend_config['remove_from_panels'] = [3]  # Panel D for 4-panel
        else:
            legend_config['remove_from_panels'] = [2]  # Panel C for 3-panel
    
    # Apply styling
    styled_fig = apply_publication_styling_to_figure(
        fig=fig,
        journal=journal,
        panel_labels=True,
        remove_titles=True,
        legend_config=legend_config,
        spine_config='left_bottom',
        **kwargs
    )
    
    # Save figure if path provided
    if save_path:
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Get journal-specific DPI if not specified
        if dpi is None:
            journal_styles = {
                'nature': 300,
                'science': 300, 
                'plos': 600,     # PLOS prefers higher resolution
                'cell': 300,
                'elife': 300,
                'biorxiv': 300   # Good balance for online viewing and file size
            }
            dpi = journal_styles.get(journal, 300)
        
        # Save in requested formats
        for fmt in save_formats:
            save_file = f"{save_path}.{fmt}"
            styled_fig.savefig(
                save_file,
                format=fmt,
                dpi=dpi,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            print(f"Saved {journal.upper()} figure: {save_file}")
    
    return styled_fig


def create_color_legend(colors: Dict[str, str],
                       ax: Optional[Axes] = None,
                       title: Optional[str] = None,
                       loc: str = 'best',
                       ncol: int = 1,
                       **kwargs) -> None:
    """
    Create a color legend from a dictionary of labels and colors.
    
    Parameters
    ----------
    colors : dict
        Dictionary mapping labels to color codes
    ax : Axes, optional
        Axes to add legend to. If None, uses current axes
    title : str, optional
        Legend title
    loc : str, default='best'
        Legend location
    ncol : int, default=1
        Number of columns in legend
    **kwargs : dict
        Additional arguments passed to ax.legend()
    """
    if ax is None:
        ax = plt.gca()
    
    patches = [mpatches.Patch(color=color, label=label) 
               for label, color in colors.items()]
    
    legend_kwargs = {
        'handles': patches,
        'title': title,
        'loc': loc,
        'ncol': ncol,
        'frameon': kwargs.get('frameon', False),
        'fontsize': kwargs.get('fontsize', plt.rcParams['font.size'])
    }
    legend_kwargs.update(kwargs)
    
    ax.legend(**legend_kwargs)


# Example usage documentation
__doc__ += """

Examples
--------
Basic usage with context manager:

>>> with publication_style('nature'):
...     fig, ax = plt.subplots(figsize=(3.35, 2.5))
...     ax.plot(x, y)
...     ax.set_xlabel('Time (s)')
...     ax.set_ylabel('Signal (mV)')
...     plt.savefig('figure1.pdf')

Creating a multi-panel figure:

>>> # Create a 2x2 figure for Science
>>> styler, fig, axes = create_multipanel_figure(4, 'square', journal='science')
>>> 
>>> for i, ax in enumerate(axes.flat):
...     # Plot data
...     ax.plot(x[i], y[i], color=styler.get_colors(1)[0])
...     
...     # Apply consistent styling
...     styler.apply_style(ax)
...     styler.format_axis(ax, 
...                       x_label='Time (ms)',
...                       y_label='Response',
...                       x_lim=(0, 100))
>>> 
>>> # Save in multiple formats
>>> styler.save_figure(fig, 'figure2', formats=['pdf', 'png', 'svg'])

Using the PublicationFigureStyler class directly:

>>> # Initialize styler for Nature
>>> styler = PublicationFigureStyler('nature')
>>> 
>>> # Create figure with custom layout
>>> fig, (ax1, ax2) = styler.setup_figure(1, 2, height_ratios=[1, 0.5])
>>> 
>>> # Plot data with journal-appropriate colors
>>> colors = styler.get_colors(5, colorblind_safe=True)
>>> for i, color in enumerate(colors):
...     ax1.plot(x, y[i], color=color, label=f'Series {i+1}')
>>> 
>>> # Apply styling and add elements
>>> styler.apply_style(ax1, spine_position='left_bottom')
>>> styler.add_panel_label(ax1, 'A')
>>> styler.add_legend(ax1, location='upper right', n_cols=2)
>>> styler.add_scale_bar(ax1, length=10, label='10 ms')
>>> 
>>> # Format axes
>>> styler.format_axis(ax1,
...                   x_label='Time (ms)',
...                   y_label='Amplitude (μV)',
...                   x_lim=(0, 100),
...                   y_lim=(-50, 50))
>>> 
>>> # Add significance bars
>>> add_significance_bar(ax1, 20, 40, 45, p_value=0.01)
>>> add_significance_bar(ax1, 60, 80, 45, p_value=0.001)
>>> 
>>> # Save figure
>>> styler.save_figure(fig, 'figure3', dpi=600)

Creating a figure with colorblind-friendly palette:

>>> with publication_style('plos'):
...     fig, ax = plt.subplots()
...     
...     # Get colorblind-friendly colors
...     colors = list(COLORBLIND_FRIENDLY.values())[:5]
...     
...     # Create grouped bar plot
...     x = np.arange(4)
...     width = 0.15
...     
...     for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
...         ax.bar(x + i*width, data, width, color=color, label=label)
...     
...     # Style the plot
...     ax.set_xticks(x + width * 2)
...     ax.set_xticklabels(['Group 1', 'Group 2', 'Group 3', 'Group 4'])
...     ax.set_ylabel('Values')
...     ax.legend()
...     
...     plt.tight_layout()
...     plt.savefig('figure4.pdf')
"""