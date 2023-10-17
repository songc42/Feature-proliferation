from ss_utils.SG2 import SG2_Generation, model_dic
from ss_utils.SG_utils import show_mul_rows, compute_z
import click
"""Generate images using pretrained network pickle."""
#----------------------------------------------------------------------------

@click.command()
@click.option('--seed', type=int, help='a tuple of random seeds', required=True)
@click.option('--truncation', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--dataset', help='what dataset pretrained model trains on', type=str, required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def SG2_modify(
    seed: int,
    truncation_psi: float,
    dataset: str,
    outdir: str
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated FFHQ images, other curated seeds for FFHQ: [70033, 79614, 70223, 70344, 79773, 79828, 79058, 3341009, 70153]
    python SG2_modify.py --outdir=out \
        --seed 70383 \
        --dataset FFHQ



    """
    
    # Determine dataset of model
    model = 'modify_'+dataset

    # Define a variable to append all images
    img_all = []
    # SG2 Generation
    img, dic_para = SG2_Generation(model, seed, operation="SG2")
    img_all.append(img)
    # SG2 modify Generation
    img_modify, dic_para = SG2_Generation(model, seed, operation="SG2_modify")
    img_all.append(img_modify)
    # SG2 Generation with Truncation Trick
    img_truncation, dic_para = SG2_Generation(model, seed, operation="SG2", truncation_psi=truncation_psi)
    img_all.append(img_truncation)

    # Show all images and save
    save_name = outdir+'/'+str(seed)+'_'+dataset+'.jpg'
    show_mul_rows(img_all, 1, save_name=save_name)




#----------------------------------------------------------------------------

if __name__ == "__main__":
    SG2_modify() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
