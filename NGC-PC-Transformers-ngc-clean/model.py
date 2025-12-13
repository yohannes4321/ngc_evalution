import jax
# from ngclearn.utils.model_utils import scanner
# from ngcsimlib.compilers import compile_command, wrap_command
# from ngcsimlib.context import Context
from ngclearn import Context, MethodProcess, JointProcess
from ngcsimlib._src.utils.io import make_unique_path, make_safe_filename
# from ngclearn.utils import JaxProcess
from ngclearn.utils.io_utils import makedir
from jax import numpy as jnp, random, jit
from ngclearn.components import GaussianErrorCell as ErrorCell, RateCell, HebbianSynapse, StaticSynapse
# import ngclearn.utils.weight_distribution as dist
from ngclearn.utils.distribution_generator import DistributionGenerator as dist
from config import Config as config
from layers.embedding import EMBEDDING
from layers.attention import Attention
from layers.blocks import Block
from utils.attention_utils import AttentionBlock
from utils.embed_utils import EmbeddingSynapse
from layers.mlp import MLP
from layers.output import Output
from utils.model_util import ReshapeComponent
from projection.projection import Projection
import numpy as np
import re 
class NGCTransformer:
    """
    Predictive Coding Transformer following PCN architecture from:
    Whittington & Bogacz (2017) - "An approximation of the error backpropagation 
    algorithm in a predictive coding network with local hebbian synaptic plasticity"

    Architecture:
    z_embed -(W_embed)-> e_embed, z_qkv -(W_q,W_k,W_v - > W_attn_out)-> e_attn, z_mlp -(W_mlp1,W_mlp2)-> e_mlp, z_out -(W_out)-> e_out
    e_attn -(E_attn)-> z_qkv <- e_embed, e_mlp -(E_mlp)-> z_mlp <- e_attn, e_out -(E_out)-> z_out <- e_mlp

    Args:
        dkey: JAX seeding key
        vocab_size: vocabulary size
        seq_len: sequence length
        n_embed: embedding dimension
        n_heads: number of attention heads
        batch_size: batch size
        n_layers: number of transformer blocks
        dt: integration time constant
        tau_m: membrane time constant
        eta: learning rate for Hebbian synapses
        exp_dir: experimental directory
        model_name: unique model name
    """

    # def __init__(self, dkey, batch_size, seq_len, n_embed, vocab_size, n_layers, n_heads,  T,
    #              dt, tau_m, act_fx, eta, exp_dir,
    #              model_name, loadDir, pos_learnable, optim_type, wlb, wub , dropout_rate, **kwargs):
    # class NGCTransformer:
    def __init__(
        self,
        dkey,                     # positional or keyword
        batch_size,               # positional or keyword
        seq_len,
        n_embed,
        vocab_size,
        n_layers,
        n_heads,
        T,
        dt,
        tau_m,
        act_fx,
        eta,
        dropout_rate,
        exp_dir,
        model_name,
        loadDir=None,             # optional keyword
        pos_learnable=False,      # optional keyword
        optim_type="adam",        # optional keyword
        wub=1.0,                  # optional keyword
        wlb=0.0,                  # optional keyword
        **kwargs                  # catch-all for future params
    ):

        self.exp_dir = exp_dir
        self.model_name = model_name
        self.nodes = None
        self.n_layers = n_layers
        self.T = T
        makedir(exp_dir)
        makedir(exp_dir + "/filters")

        dkey, *subkeys = random.split(dkey, 50)
        print("🔥 ENTERED __init__ 🔥", flush=True)
        print("loadDir =", loadDir, flush=True)

        if loadDir is not None:
            print("➡️ Calling load_from_disk()", flush=True)
            self.load_from_disk(loadDir)
            print("⬅️ Returned from load_from_disk()", flush=True)
        elif loadDir is not None:
            print("❌ loadDir is None — NOT loading from disk", flush=True)

        else:
            with Context("Circuit") as self.circuit:
                
                self.embedding = EMBEDDING(dkey=subkeys[0], vocab_size=vocab_size, seq_len=seq_len, embed_dim=n_embed, batch_size=batch_size, pos_learnable=pos_learnable, eta=eta, optim_type=optim_type)
                
                self.blocks = []
                for i in range(n_layers):
                    key, subkey = random.split(subkeys[1 + i])
                    block=Block(dkey=subkey, block_id= i, n_embed=n_embed, seq_len=seq_len,
                                batch_size=batch_size, vocab_size=vocab_size, n_heads=n_heads, dropout_rate=dropout_rate, eta=eta, optim_type=optim_type, wub=wub, wlb=wlb, tau_m=tau_m)
                    self.blocks.append(block)   
                    
                self.output = Output(dkey=subkeys[3], n_embed=n_embed, seq_len=seq_len, batch_size=batch_size, vocab_size=vocab_size, eta=eta, optim_type=optim_type, wlb=wlb, wub=wub, tau_m=tau_m)
                
                self.z_target=RateCell("z_target", n_units= vocab_size, tau_m=0., act_fx="identity", batch_size=batch_size * seq_len) 
                self.z_actfx= RateCell("z_actfx", n_units= vocab_size, tau_m=0., act_fx="softmax", batch_size=batch_size * seq_len)
                
                self.reshape_4d_to_2d = ReshapeComponent("reshape_4d_to_2d",
                                            input_shape=(batch_size, seq_len, n_embed, 1),
                                            output_shape=(batch_size * seq_len, n_embed))
                
                self.reshape_3d_to_2d_embed = ReshapeComponent("reshape_3d_to_2d_embed",
                                            input_shape=(batch_size, seq_len, n_embed),
                                            output_shape=(batch_size * seq_len, n_embed))
                self.reshape_2d_to_3d_embed= ReshapeComponent("reshape_2d_to_3d_embed",
                                            input_shape=(batch_size * seq_len, n_embed),
                                            output_shape=(batch_size, seq_len, n_embed))
                
                
                self.embedding.W_embed.inputs >> self.embedding.z_embed.zF  
                self.reshape_3d_to_2d_embed.inputs >> self.embedding.W_embed.outputs   
                self.embedding.e_embed.mu >> self.reshape_3d_to_2d_embed.outputs
                self.embedding.e_embed.target >> self.blocks[0].attention.z_qkv.z
                
                # self.reshape_4d_to_2d.inputs >> self.attention.z_qkv.zF
                for blocks in range(n_layers):
                    block= self.blocks[blocks]
                    block.attention.z_qkv.zF  >>  block.attention.W_q.inputs
                    block.attention.z_qkv.zF >>   block.attention.W_k.inputs 
                    block.attention.z_qkv.zF >>   block.attention.W_v.inputs
                    
                    block.attention.W_q.outputs >> block.reshape_2d_to_3d_q.inputs 
                    block.attention.W_k.outputs >> block.reshape_2d_to_3d_k.inputs 
                    block.attention.W_v.outputs >> block.reshape_2d_to_3d_v.inputs 
                    
                    block.reshape_2d_to_3d_q.outputs >> block.attention.attn_block.inputs_q
                    block.reshape_2d_to_3d_k.outputs >> block.attention.attn_block.inputs_k
                    block.reshape_2d_to_3d_v.outputs >> block.attention.attn_block.inputs_v
                    block.attention.attn_block.outputs >> block.reshape_3d_to_2d.inputs

                    block.reshape_3d_to_2d.outputs >> block.attention.W_attn_out.inputs
                    block.attention.W_attn_out.outputs >> block.attention.e_attn.mu
                    block.mlp.z_mlp.z >> block.attention.e_attn.target


                    block.mlp.z_mlp.zF >> block.mlp.W_mlp1.inputs
                    block.mlp.W_mlp1.outputs >> block.mlp.e_mlp1.mu
                    block.mlp.z_mlp2.z >> block.mlp.e_mlp1.target


                    block.mlp.z_mlp2.zF >> block.mlp.W_mlp2.inputs
                    block.mlp.W_mlp2.outputs >> block.mlp.e_mlp.mu

     
                    
                    if blocks == n_layers - 1:
                        self.output.z_out.z >> block.mlp.e_mlp.target
                    else:
                        self.blocks[blocks + 1].attention.z_qkv.z >> block.mlp.e_mlp.target


                    block.attention.e_attn.dmu >> block.attention.E_attn.inputs

                    block.mlp.e_mlp1.dmu >> block.mlp.E_mlp1.inputs
                    block.mlp.e_mlp.dmu  >> block.mlp.E_mlp.inputs

                    block.attention.E_attn.outputs >> block.attention.z_qkv.j


                    if blocks == 0:
                        self.embedding.e_embed.dtarget >> block.attention.z_qkv.j_td
                    else:
                        block.mlp.e_mlp.dtarget >> block.attention.z_qkv.j_td


                    block.mlp.E_mlp.outputs  >> block.mlp.z_mlp2.j
                    block.mlp.E_mlp1.outputs >> block.mlp.z_mlp.j

                    block.attention.e_attn.dtarget >> block.mlp.z_mlp.j_td
                    block.mlp.e_mlp1.dtarget       >> block.mlp.z_mlp2.j_td


                    block.attention.z_qkv.zF >> block.attention.W_q.pre
                    block.attention.e_attn.dmu >> block.attention.W_q.post

                    block.attention.z_qkv.zF >> block.attention.W_k.pre
                    block.attention.e_attn.dmu >> block.attention.W_k.post

                    block.attention.z_qkv.zF >> block.attention.W_v.pre
                    block.attention.e_attn.dmu >> block.attention.W_v.post


                    block.attention.attn_block.outputs >> block.reshape_3d_to_2d_attnout.inputs
                    block.reshape_3d_to_2d_attnout.outputs >> block.attention.W_attn_out.pre
                    block.attention.e_attn.dmu >> block.attention.W_attn_out.post


                    block.mlp.z_mlp.zF  >> block.mlp.W_mlp1.pre
                    block.mlp.e_mlp1.dmu >> block.mlp.W_mlp1.post

                    block.mlp.z_mlp2.zF >> block.mlp.W_mlp2.pre
                    block.mlp.e_mlp.dmu  >> block.mlp.W_mlp2.post

                        
                self.output.z_out.zF >> self.output.W_out.inputs
                self.output.W_out.outputs >> self.z_actfx.j

                self.z_actfx.zF >> self.output.e_out.mu
                self.z_target.z >> self.output.e_out.target

                self.output.e_out.dmu >> self.output.E_out.inputs


                self.output.E_out.outputs >> self.output.z_out.j
                self.blocks[n_layers - 1].mlp.e_mlp.dtarget >> self.output.z_out.j_td


                self.embedding.e_embed.dmu >> self.reshape_2d_to_3d_embed.inputs
                self.reshape_2d_to_3d_embed.outputs >> self.embedding.W_embed.post


                self.output.z_out.zF >> self.output.W_out.pre
                self.output.e_out.dmu >> self.output.W_out.post

                        
                        
                ## PROJECTION PHASE ##
                
                self.projection = Projection(dkey=subkeys[29], n_embed=n_embed, seq_len=seq_len, batch_size=batch_size,
                                             vocab_size=vocab_size, eta=eta, optim_type=optim_type, pos_learnable=pos_learnable, wub=wub, wlb=wlb, n_blocks=n_layers, n_heads=n_heads, dropout_rate=dropout_rate)
                
                
                self.projection.q_embed_Ratecell.zF >> self.projection.Q_embed.inputs
                self.projection.Q_embed.outputs >> self.projection.reshape_3d_to_2d_proj.inputs

                for b in range(n_layers):
                    block_proj = self.projection.blocks[b]

                    if b == 0:
                        self.projection.reshape_3d_to_2d_proj.outputs >> block_proj.q_qkv_Ratecell.j
                    else:
                        self.projection.blocks[b - 1].Q_mlp2.outputs >> block_proj.q_qkv_Ratecell.j

                    block_proj.q_qkv_Ratecell.zF >> block_proj.Q_q.inputs
                    block_proj.q_qkv_Ratecell.zF >> block_proj.Q_k.inputs
                    block_proj.q_qkv_Ratecell.zF >> block_proj.Q_v.inputs

                    block_proj.Q_q.outputs >> block_proj.q_attn_block.inputs_q
                    block_proj.Q_k.outputs >> block_proj.q_attn_block.inputs_k
                    block_proj.Q_v.outputs >> block_proj.q_attn_block.inputs_v

                    block_proj.q_attn_block.outputs >> block_proj.reshape_3d_to_2d_proj1.inputs
                    block_proj.reshape_3d_to_2d_proj1.outputs >> block_proj.Q_attn_out.inputs
                    block_proj.Q_attn_out.outputs >> block_proj.q_mlp_Ratecell.j

                    block_proj.q_mlp_Ratecell.zF >> block_proj.Q_mlp1.inputs
                    block_proj.Q_mlp1.outputs >> block_proj.q_mlp2_Ratecell.j
                    block_proj.q_mlp2_Ratecell.zF >> block_proj.Q_mlp2.inputs

                self.projection.blocks[n_layers - 1].Q_mlp2.outputs >> self.projection.q_out_Ratecell.j
                self.projection.q_out_Ratecell.zF >> self.projection.Q_out.inputs
                self.projection.Q_out.outputs >> self.projection.q_target_Ratecell.j

                self.projection.q_target_Ratecell.z >> self.projection.eq_target.mu

                
                # Create the processes by iterating through all blocks
                advance_process = MethodProcess(name="advance_process")
                                   
                                   
                                   
                                   
                    
                reset_process = MethodProcess(name="reset_process")
                embedding_evolve_process = MethodProcess(name="embedding_evolve_process")
                                           
                evolve_process = MethodProcess(name="evolve_process")
                project_process = MethodProcess(name="project_process")
                embedding_evolve_process  >> self.embedding.W_embed.evolve



                advance_process >> self.embedding.z_embed.advance_state
                advance_process >> self.embedding.W_embed.advance_state
                advance_process >> self.reshape_3d_to_2d_embed.advance_state
                advance_process >> self.reshape_2d_to_3d_embed.advance_state
                advance_process >> self.embedding.e_embed.advance_state

                for i in range(n_layers):
                    block = self.blocks[i]
                    
                    advance_process >> block.attention.E_attn.advance_state
                    advance_process >> block.mlp.E_mlp.advance_state
                    advance_process >> block.attention.z_qkv.advance_state
                    advance_process >> block.mlp.z_mlp.advance_state
                    advance_process >> block.mlp.z_mlp2.advance_state
                    advance_process >> block.attention.W_q.advance_state
                    advance_process >> block.attention.W_k.advance_state
                    advance_process >> block.attention.W_v.advance_state
                    advance_process >> block.reshape_2d_to_3d_q.advance_state
                    advance_process >> block.reshape_2d_to_3d_k.advance_state
                    advance_process >> block.reshape_2d_to_3d_v.advance_state
                    advance_process >> block.attention.attn_block.advance_state
                    advance_process >> block.reshape_3d_to_2d.advance_state
                    advance_process >> block.reshape_3d_to_2d_attnout.advance_state
                    advance_process >> block.attention.W_attn_out.advance_state
                    advance_process >> block.mlp.W_mlp1.advance_state
                    advance_process >> block.mlp.W_mlp2.advance_state
                    advance_process >> block.attention.e_attn.advance_state
                    advance_process >> block.mlp.e_mlp.advance_state
                    
                    reset_process >> block.attention.z_qkv.reset
                    reset_process >> block.mlp.z_mlp.reset
                    reset_process >> block.mlp.z_mlp2.reset
                    reset_process >> block.attention.e_attn.reset
                    reset_process >> block.mlp.e_mlp.reset
                    reset_process >> block.mlp.e_mlp1.reset
                    reset_process >> block.reshape_3d_to_2d.reset
                    reset_process >> block.reshape_2d_to_3d_q.reset
                    reset_process >> block.reshape_2d_to_3d_k.reset
                    reset_process >> block.reshape_2d_to_3d_v.reset
                    reset_process >> block.reshape_3d_to_2d_attnout.reset
                    
                    evolve_process >> block.attention.W_q.evolve
                    evolve_process >> block.attention.W_k.evolve
                    evolve_process >> block.attention.W_v.evolve
                    evolve_process >> block.attention.W_attn_out.evolve
                    evolve_process >> block.mlp.W_mlp1.evolve
                    evolve_process >> block.mlp.W_mlp2.evolve

                # Add non-block components to advance_process, reset_process, evolve_process
                advance_process >> self.output.E_out.advance_state
                advance_process >> self.output.z_out.advance_state
                advance_process >> self.output.W_out.advance_state
                advance_process >> self.z_actfx.advance_state
                advance_process >> self.output.e_out.advance_state

                reset_process >> self.projection.q_embed_Ratecell.reset
                reset_process >> self.projection.q_out_Ratecell.reset
                reset_process >> self.projection.q_target_Ratecell.reset
                reset_process >> self.projection.eq_target.reset
                reset_process >> self.embedding.z_embed.reset
                reset_process >> self.output.z_out.reset
                reset_process >> self.z_target.reset
                reset_process >> self.z_actfx.reset
                reset_process >> self.embedding.e_embed.reset
                reset_process >> self.output.e_out.reset
                reset_process >> self.reshape_3d_to_2d_embed.reset
                reset_process >> self.reshape_2d_to_3d_embed.reset

                evolve_process >> self.output.W_out.evolve
                project_process >> self.projection.q_embed_Ratecell.advance_state
                project_process >> self.projection.Q_embed.advance_state
                project_process >> self.projection.reshape_3d_to_2d_proj.advance_state
                for b in range(n_layers):
                    block_proj= self.projection.blocks[b]
                    project_process >> block_proj.q_qkv_Ratecell.advance_state
                    project_process >> block_proj.Q_q.advance_state
                    project_process >> block_proj.Q_k.advance_state
                    project_process >> block_proj.Q_v.advance_state
                    project_process >> block_proj.q_attn_block.advance_state
                    project_process >> block_proj.reshape_3d_to_2d_proj1.advance_state
                    project_process >> block_proj.Q_attn_out.advance_state
                    project_process >> block_proj.q_mlp_Ratecell.advance_state
                    project_process >> block_proj.q_mlp2_Ratecell.advance_state
                    project_process >> block_proj.Q_mlp1.advance_state
                    project_process >> block_proj.Q_mlp2.advance_state
                    reset_process >> block_proj.q_qkv_Ratecell.reset
                    reset_process >> block_proj.q_attn_block.reset
                    reset_process >> block_proj.q_mlp_Ratecell.reset
                    reset_process >> block_proj.q_mlp2_Ratecell.reset 
                project_process >> self.projection.q_out_Ratecell.advance_state
                project_process >> self.projection.Q_out.advance_state
                project_process >> self.projection.q_target_Ratecell.advance_state
                project_process >> self.projection.eq_target.advance_state
                
                # processes = (reset_process, advance_process, embedding_evolve_process, evolve_process, project_process)
                self.reset = reset_process
                self.advance = advance_process
                self.evolve = evolve_process
                self.project = project_process
                self.embedding_evolve=embedding_evolve_process

                


    # @Context.dynamicCommand
    def clamp_input(self,x):
        self.embedding.z_embed.j.set(x)
        self.projection.q_embed_Ratecell.j.set(x) 
        
    # @Context.dynamicCommand
    def clamp_target(self,y):
        self.z_target.j.set(y)

    # @Context.dynamicCommand
    def clamp_infer_target(self,y):
        self.projection.eq_target.target.set(y)
        
    def save_to_disk(self, params_only=False):
        """
        Saves current model parameter get()s to disk

        Args:
            params_only: if True, save only param arrays to disk (and not JSON sim/model structure)
        """
        if params_only:
            model_dir = "{}/{}/component/custom".format(self.exp_dir, self.model_name)
            self.embedding.W_embed.save(model_dir)
            self.blocks = []
            for j in range(self.n_layers):
                block = self.circuit.get_components(f"block{j}_W_q")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_k")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_v")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_attn_out")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_mlp1")
                block.save(model_dir)
                block = self.circuit.get_components(f"block{j}_W_mlp2")
                block.save(model_dir)    
            self.output.W_out.save(model_dir)
        else:
            self.circuit.save_to_json(self.exp_dir, model_name=self.model_name, overwrite=True)
            
    # def load_from_disk(self, model_directory):
    #     """
    #     Loads parameter/config get()s from disk to this model

    #     Args:
    #         model_directory: directory/path to saved model parameter/config get()s
    #     """
    #     self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
        
    #     # Load processes
    #     processes = self.circuit.get_objects_by_type("process")
    #     self.advance = processes.get("advance_process")
    #     self.reset   = processes.get("reset_process")
    #     self.evolve  = processes.get("evolve_process")
    #     self.project = processes.get("project_process")

    #     # Load all neural network components by their exact names
    #     nodes = self.circuit.get_components(
    #         "q_embed_Ratecell", "q_out_Ratecell", "q_target_Ratecell", "q_qkv_Ratecell", "q_mlp_Ratecell", "q_mlp2_Ratecell", "q_attn_block", "eq_target",
    #         "Q_q", "Q_k", "Q_v", "Q_attn_out", "Q_mlp1", "Q_mlp2", "Q_embed", "Q_out",
    #         "z_embed", "z_qkv", "z_mlp", "z_mlp2", "z_out",
    #         "e_embed", "e_attn", "e_mlp", "e_mlp1", "e_out",
    #         "W_embed", "W_q", "W_k", "W_v", "W_attn_out", "W_mlp1", "W_mlp2", "W_out",
    #         "E_attn", "E_mlp1", "E_mlp", "E_out"
    #     )

    #     # Unpack with meaningful, descriptive names
    #     (self.q_embed_Ratecell, self.q_out_Ratecell, self.q_target_Ratecell, self.q_qkv, self.q_mlp_Ratecell, self.q_mlp2_Ratecell,
    #     self.q_attn_block, self.eq_target,
    #     self.Q_q, self.Q_k, self.Q_v, self.Q_attn_out, self.Q_mlp1, self.Q_mlp2, self.Q_embed, self.Q_out,
    #     self.z_embed, self.z_qkv, self.z_mlp, self.z_mlp2, self.z_out,
    #     self.e_embed, self.e_attn, self.e_mlp, self.e_mlp1, self.e_out,
    #     self.W_embed, self.W_q, self.W_k, self.W_v, self.W_attn_out,
    #     self.W_mlp1, self.W_mlp2, self.W_out,
    #     self.E_attn, self.E_mlp1, self.E_mlp, self.E_out) = nodes

    # def load_from_disk(self, model_directory):
    #     """
    #     Loads parameters/configs from disk to this model
    #     and also makes the raw weights accessible.
    #     """
    #     print("🔄 Loading model from disk...")
    #     print(f"📁 Model directory: {model_directory}")

    #     # Load the context
    #     self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
    #     print("✅ Context loaded successfully")
    #     # ---------------------------------------------------------
    #     # 🟢 PASTE THIS DEBUG BLOCK HERE 🟢
    #     # This will print every valid name in your loaded model
    #     # so you can spot if it is named "W_mlp1" or just "W_mlp"
    #     # ---------------------------------------------------------
    #     all_components = self.circuit.get_objects_by_type("component")
    #     print("\n📝 --- DEBUG: AVAILABLE COMPONENT NAMES ---")
    #     for name in all_components.keys():
    #         print(f"   • {name}")
    #     print("-------------------------------------------\n")
    #     # ---------------------------------------------------------

    #     # Load processes
    #     processes = self.circuit.get_objects_by_type("process")
    #     self.advance = processes.get("advance_process")
    #     self.reset   = processes.get("reset_process")
    #     self.evolve  = processes.get("evolve_process")
    #     self.project = processes.get("project_process")

    #     # Load components
    #     component_names = (
    #         "q_embed_Ratecell", "q_out_Ratecell", "q_target_Ratecell",
    #         "q_qkv_Ratecell", "q_mlp_Ratecell", "q_mlp2_Ratecell",
    #         "q_attn_block", "eq_target",
    #         "Q_q", "Q_k", "Q_v", "Q_attn_out",
    #         "Q_mlp1", "Q_mlp2", "Q_embed", "Q_out",
    #         "z_embed", "z_qkv", "z_mlp", "z_mlp2", "z_out",
    #         "e_embed", "e_attn", "e_mlp", "e_mlp1", "e_out",
    #         "W_embed", "W_q", "W_k", "W_v", "W_attn_out",
    #         "W_mlp1", "W_mlp2", "W_out",
    #         "E_attn", "E_mlp1", "E_mlp", "E_out"
    #     )

    #     nodes = self.circuit.get_components(*component_names)
        
    #     # Unpack components
    #     (
    #         self.q_embed_Ratecell, self.q_out_Ratecell, self.q_target_Ratecell,
    #         self.q_qkv, self.q_mlp_Ratecell, self.q_mlp2_Ratecell,
    #         self.q_attn_block, self.eq_target,
    #         self.Q_q, self.Q_k, self.Q_v, self.Q_attn_out,
    #         self.Q_mlp1, self.Q_mlp2, self.Q_embed, self.Q_out,
    #         self.z_embed, self.z_qkv, self.z_mlp, self.z_mlp2, self.z_out,
    #         self.e_embed, self.e_attn, self.e_mlp, self.e_mlp1, self.e_out,
    #         self.W_embed, self.W_q, self.W_k, self.W_v, self.W_attn_out,
    #         self.W_mlp1, self.W_mlp2, self.W_out,
    #         self.E_attn, self.E_mlp1, self.E_mlp, self.E_out
    #     ) = nodes

    #     print("✅ Components loaded successfully")


        # -------------------------------
        

                # print(nodes)
    # Import the regular expression module for string parsing

#     import re 
# from ngcsimlib.context import Context 

    def load_from_disk(self, model_directory):
        """
        Loads parameters/configs from disk to this model, 
        dynamically targeting the components of the final block (block3).
        """
        print("🔄 Loading model from disk...")
        print(f"📁 Model directory: {model_directory}")

        # Load the context
        self.circuit = Context.load(directory=model_directory, module_name=self.model_name)
        print("✅ Context loaded successfully")
        
        # ---------------------------------------------------------
        # 1. FIND THE FINAL BLOCK INDEX
        # ---------------------------------------------------------
        all_components = self.circuit.get_objects_by_type("component")
        
        block_indices = []
        # Regex to find block number in 'blockX_' or 'proj_blockX_'
        block_pattern = re.compile(r"(?:proj_)?block(\d+)_") 
        
        for name in all_components.keys():
            match = block_pattern.match(name)
            if match:
                block_indices.append(int(match.group(1)))
        
        if not block_indices:
            FINAL_BLOCK_IDX = 0 
        else:
            FINAL_BLOCK_IDX = max(block_indices)
        
        FINAL_BLOCK_PREFIX = f"block{FINAL_BLOCK_IDX}_"
        PROJ_FINAL_BLOCK_PREFIX = f"proj_block{FINAL_BLOCK_IDX}_"
        print(f"🎯 Determined Final Block Index: {FINAL_BLOCK_IDX}")
        print(f"🎯 Using Main Block Prefix: {FINAL_BLOCK_PREFIX}")
        # ---------------------------------------------------------

        # Load processes (These names are constant)
        processes = self.circuit.get_objects_by_type("process")
        self.advance = processes.get("advance_process")
        self.reset   = processes.get("reset_process")
        self.evolve  = processes.get("evolve_process")
        self.project = processes.get("project_process")

        # ---------------------------------------------------------
        # 2. DEFINE COMPONENTS TO LOAD
        # ---------------------------------------------------------
        
        def get_final_name(base_var_name):
            # Top-level components (No prefix)
            top_level_names = ["Q_embed", "Q_out", "W_embed", "W_out", "z_embed", "z_out", 
                            "e_embed", "e_out", "eq_target", "q_embed_Ratecell", "q_out_Ratecell", 
                            "q_target", "E_out"]
            if base_var_name in top_level_names:
                # Note: For the single q_target, we must use 'q_target' and NOT 'q_target_Ratecell'
                if base_var_name == "q_target_Ratecell":
                    return "q_target"
                return base_var_name

            # Projection block components (Uses PROJ_FINAL_BLOCK_PREFIX)
            # Components starting with q_ or Q_ (that are not top-level) are in proj_blockX_
            if base_var_name.startswith("q_") or base_var_name.startswith("Q_"):
                # e.g., 'Q_q' -> 'proj_block3_Q_q'
                return f"{PROJ_FINAL_BLOCK_PREFIX}{base_var_name}"
            
            # Main block components (Uses FINAL_BLOCK_PREFIX)
            # Components starting with W_, z_, e_, or E_ (that are not top-level) are in blockX_
            if base_var_name.startswith("W_") or base_var_name.startswith("z_") or base_var_name.startswith("e_") or base_var_name.startswith("E_"):
                # e.g., 'W_mlp1' -> 'block3_W_mlp1'
                return f"{FINAL_BLOCK_PREFIX}{base_var_name}"
                
            # Fallback (Should not be reached for your list)
            return base_var_name

        # List of variable names in the exact order of the unpacking tuple
        var_names = [
            "q_embed_Ratecell", "q_out_Ratecell", "q_target_Ratecell",
            "q_qkv", "q_mlp_Ratecell", "q_mlp2_Ratecell",
            "q_attn_block", "eq_target",
            "Q_q", "Q_k", "Q_v", "Q_attn_out",
            "Q_mlp1", "Q_mlp2", "Q_embed", "Q_out",
            "z_embed", "z_qkv", "z_mlp", "z_mlp2", "z_out",
            "e_embed", "e_attn", "e_mlp", "e_mlp1", "e_out",
            "W_embed", "W_q", "W_k", "W_v", "W_attn_out",
            "W_mlp1", "W_mlp2", "W_out",
            "E_attn", "E_mlp1", "E_mlp", "E_out"
        ]
        
        # Generate the component names list
        component_names = [get_final_name(var_name) for var_name in var_names]

        # ---------------------------------------------------------
        # 3. LOAD COMPONENTS ROBUSTLY
        # ---------------------------------------------------------
        nodes = []
        
        for name in component_names:
            # get_components returns a list, e.g., [component_object] or [None]
            result_list = self.circuit.get_components(name)
            
            # Check if the result is a list and the first element is a component
            if result_list and result_list[0] is not None:
                nodes.append(result_list[0])
            else:
                # Component not found (This will only happen if one of the generated names is incorrect)
                print(f"🛑 Error: Component '{name}' (required for loading) was NOT FOUND. Setting corresponding attribute to None.")
                nodes.append(None)

        # 4. UNPACK COMPONENTS (The number of elements in 'nodes' must match the number of attributes!)
        (
            self.q_embed_Ratecell, self.q_out_Ratecell, self.q_target_Ratecell,
            self.q_qkv, self.q_mlp_Ratecell, self.q_mlp2_Ratecell,
            self.q_attn_block, self.eq_target,
            self.Q_q, self.Q_k, self.Q_v, self.Q_attn_out,
            self.Q_mlp1, self.Q_mlp2, self.Q_embed, self.Q_out,
            self.z_embed, self.z_qkv, self.z_mlp, self.z_mlp2, self.z_out,
            self.e_embed, self.e_attn, self.e_mlp, self.e_mlp1, self.e_out,
            self.W_embed, self.W_q, self.W_k, self.W_v, self.W_attn_out,
            self.W_mlp1, self.W_mlp2, self.W_out,
            self.E_attn, self.E_mlp1, self.E_mlp, self.E_out
        ) = nodes

        print("✅ Components loaded successfully")
        
        # -------------------------------
        # Final Test (Accessing the weights compartment)
        # -------------------------------
        if self.W_mlp1:
            print("\n--- Weights Test (Final Block) ---")
            # Access the compartment and get the array value
            print(f"W_mlp1 (from {get_final_name('W_mlp1')}) loaded successfully.")
            
            weights_data = self.W_mlp1.weights.get()
            print(f"Shape: {weights_data.shape}, Mean: {weights_data.mean():.4f}, Max: {weights_data.max():.4f}")
            print(f"First 5x5 weights of W_mlp1:\n{weights_data[:5, :5]}")
            
            print("--------------------")
        else:
            print(f"W_mlp1 (from {get_final_name('W_mlp1')}) failed to load. Check console for 🛑 Error.")
            
            # -------------------------------
            # Load raw parameter values (Final Test)
            # -------------------------------
            # Note: self.W_mlp1 now holds the weights from the final block (block3 in your case)
        

# ... rest of your class/file ...


    def process(self, obs, lab, adapt_synapses=True):
        self.q_embed_Ratecell.get()
        eps = 0.001
        # scale = 1.0 / jnp.sqrt(config.n_embed) 
        # self.circuit.reset()
        self.reset.run()

        ## pin/tie inference synapses to be exactly equal to the forward ones
        
        self.projection.Q_embed.word_weights.set(self.embedding.W_embed.word_weights.get())
        if self.embedding.W_embed.pos_learnable:
           self.projection.Q_embed.pos_weights.set(self.embedding.W_embed.pos_weights.get())
        for i in range(self.n_layers):
            block_proj= self.projection.blocks[i]
            block= self.blocks[i] #lk
            block_proj.Q_q.weights.set(block.attention.W_q.weights.get())
            block_proj.Q_q.biases.set(block.attention.W_q.biases.get())
            block_proj.Q_k.weights.set(block.attention.W_k.weights.get())
            block_proj.Q_k.biases.set(block.attention.W_k.biases.get())
            block_proj.Q_v.weights.set(block.attention.W_v.weights.get())
            block_proj.Q_v.biases.set(block.attention.W_v.biases.get())
            block_proj.Q_attn_out.weights.set(block.attention.W_attn_out.weights.get())
            block_proj.q_attn_block.inputs_q.set(block.attention.attn_block.inputs_q.get())
            block_proj.q_attn_block.inputs_k.set(block.attention.attn_block.inputs_k.get())
            block_proj.q_attn_block.inputs_v.set(block.attention.attn_block.inputs_v.get())
            block_proj.Q_attn_out.biases.set(block.attention.W_attn_out.biases.get())
            block_proj.Q_mlp1.weights.set(block.mlp.W_mlp1.weights.get())
            block_proj.Q_mlp1.biases.set(block.mlp.W_mlp1.biases.get())
            block_proj.Q_mlp2.weights.set(block.mlp.W_mlp2.weights.get())
            block_proj.Q_mlp2.biases.set(block.mlp.W_mlp2.biases.get())
            
            ## pin/tie feedback synapses to transpose of forward ones

            block.attention.E_attn.weights.set(jnp.transpose(block.attention.W_attn_out.weights.get()))
            block.mlp.E_mlp.weights.set(jnp.transpose(block.mlp.W_mlp2.weights.get()))  
            block.mlp.E_mlp1.weights.set(jnp.transpose(block.mlp.W_mlp1.weights.get()))
  
        self.projection.Q_out.weights.set(self.output.W_out.weights.get())
        self.projection.Q_out.biases.set(self.output.W_out.biases.get())
        self.projection.q_target_Ratecell.j_td.set(jnp.zeros((config.batch_size * config.seq_len, config.vocab_size)))
        
        ## pin/tie feedback synapses to transpose of forward ones
       
        self.output.E_out.weights.set(jnp.transpose(self.output.W_out.weights.get()))
        
        ## Perform P-step (projection step)
        self.clamp_input(obs)
        self.clamp_infer_target(lab)
        self.project.run(t=0., dt=1.)
        # initialize dynamics of generative model latents to projected states for the errors it's 0
        self.blocks[0].attention.z_qkv.z.set(self.projection.blocks[0].q_qkv_Ratecell.z.get())
        self.blocks[0].mlp.z_mlp.z.set(self.projection.blocks[0].q_mlp_Ratecell.z.get())
        self.blocks[0].mlp.z_mlp2.z.set(self.projection.blocks[0].q_mlp2_Ratecell.z.get())
        self.output.z_out.z.set(self.projection.q_out_Ratecell.z.get())
        self.output.e_out.dmu.set(self.projection.eq_target.dmu.get())
        self.output.e_out.dtarget.set(self.projection.eq_target.dtarget.get())
        
        
        ## get projected prediction (from the P-step)
        y_mu_inf = self.projection.q_target_Ratecell.z.get()
    
        EFE = 0. ## expected free energy
        y_mu = 0.
        if adapt_synapses:
            for ts in range(0, self.T):
                # self.circuit.clamp_input(obs) ## clamp input data to z_embed & q_embed input compartments
                # self.circuit.clamp_target(lab) ## clamp target data to z_target
                self.clamp_input(obs)
                self.clamp_target(lab)
                # self.circuit.advance(t=ts, dt=1.)
                self.advance.run(t=ts,dt=1.)
           
        y_mu = self.output.W_out.outputs.get() ## get settled prediction

        L1 = self.embedding.e_embed.L.get()
        L4 = self.output.e_out.L.get()
            # Sum errors from ALL blocks
        block_errors = 0.
        for i in range(self.n_layers):
                block = self.blocks[i]
                block_errors += block.attention.e_attn.L.get() + block.mlp.e_mlp.L.get() + block.mlp.e_mlp1.L.get()

        EFE = L4 + block_errors + L1

        if adapt_synapses == True:
                self.embedding_evolve.run()
                self.evolve.run(t=self.T,dt=1.)
                
        ## skip E/M steps if just doing test-time inference
        return y_mu_inf, y_mu, EFE

    def get_latents(self):
        return self.q_out_Ratecell.z.get()
    
    