#!/usr/bin/env python
"""
Function dependency graph for llama3.py with actual dimensions
"""
from graphviz import Digraph

def create_function_graph():
    dot = Digraph(comment='llama3.py function dependencies')
    dot.attr(rankdir='TB')
    
    # Global settings for better resolution
    dot.attr('node', 
             shape='box', 
             style='rounded,filled', 
             fillcolor='lightgrey',
             fontname='Arial',
             fontsize='14',
             height='0.8',
             width='2.5')
    dot.attr('edge', 
             fontname='Arial',
             fontsize='12',
             penwidth='1.5')
    
    # Dimension legend with better formatting
    dot.node('legend', '''Dimensions:
B: batch_size (32)
L: seq_len (256)
D: dim (288)
HN: n_heads (6)
HD: head_dim (48)
VS: vocab_size (32000)
FD: ffn_dim (768)''')
    
    # Core components with actual dimensions
    dot.node('Llama', '''Llama
{B:32, L:256, D:288}
→
{B:32, 1, VS:32000}''')
    dot.node('TransformerBlock', '''TransformerBlock
{B:32, L:256, D:288}
→
{B:32, L:256, D:288}''')
    dot.node('Attention', '''Attention
{B:32, L:256, D:288}
→
{B:32, L:256, D:288}''')
    dot.node('FFN', '''FFN
{B:32, L:256, D:288}
→
{B:32, L:256, D:288}''')
    dot.node('RMSNorm', '''RMSNorm
{B:32, L:256, D:288}
→
{B:32, L:256, D:288}''')
    
    # Operations with actual dimensions
    dot.node('softmax', '''softmax
{B:32, HN:6, L:256, L:256}
→
{B:32, HN:6, L:256, L:256}''')
    dot.node('silu', '''silu
{B:32, L:256, FD:768}
→
{B:32, L:256, FD:768}''')
    dot.node('rope', '''RoPE
{B:32, L:256, HN:6, HD:48}
→
{B:32, L:256, HN:6, HD:48}''')
    dot.node('repeat_kv', '''repeat_kv
{B:32, L:256, KVHN:6, HD:48}
→
{B:32, L:256, HN:6, HD:48}''')
    
    # Dependencies with consistent edge labels
    dot.edge('legend', 'Llama', 'defines')
    
    # Model architecture edges
    dot.edge('Llama', 'TransformerBlock', 'contains [n=6]')
    dot.edge('Llama', 'RMSNorm', 'contains')
    dot.edge('TransformerBlock', 'Attention', 'contains')
    dot.edge('TransformerBlock', 'FFN', 'contains')
    dot.edge('TransformerBlock', 'RMSNorm', 'contains [n=2]')
    
    # Operation edges
    dot.edge('Attention', 'rope', 'applies (dim: HD=D/HN)')
    dot.edge('Attention', 'repeat_kv', 'applies')
    dot.edge('Attention', 'softmax', 'applies')
    dot.edge('FFN', 'silu', 'applies (dim: FD=2*4*D/3)')
    
    # Generation flow
    dot.node('generate', '''generate
{B:32, L:256}
→
{B:32, 1}''')
    dot.edge('Llama', 'generate', 'implements')
    dot.edge('generate', 'Llama.__call__', 'calls')
    
    # Layout settings for better resolution
    dot.attr(size='11,16')  # Larger page size
    dot.attr(ratio='compress')  # Allow graph to expand
    dot.attr(concentrate='true')  # Merge edges
    dot.attr(ranksep='2.0')  # More vertical space between ranks
    dot.attr(nodesep='1.5')  # More horizontal space between nodes
    dot.attr(dpi='300')  # Higher DPI for better resolution
    
    # Render with high DPI and better quality
    dot.render('llama3_dims', format='png', cleanup=True)

if __name__ == '__main__':
    create_function_graph() 