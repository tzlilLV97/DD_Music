from frechet_audio_distance import FrechetAudioDistance
import numpy as np

scores = np.array([{}])

# to use `vggish`
frechet_vggish = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False
)

# to use `PANN`
frechet_PANN = FrechetAudioDistance(
    model_name="pann",
    sample_rate=16000,
    use_pca=False,
    use_activation=False,
    verbose=False
)

fad = [frechet_vggish, frechet_PANN]

path_to_background_set = r"C:\path-to-your-BACKGROUND-audio-files-folder"
path_to_eval_set = r"C:\path-to-your-GENERATED-audio-files-folder"

# Specify the paths to your saved embeddings
# background_embds_path = r"C:\Projects\frechet-audio-distance\our\Nlakh_embeddings_"
background_embds_path = r"C:\path-to-save-the-BACKGROUND-files-distribution\Background_embeddings_"
eval_embds_path = r"C:\path-to-save-the-GENERATED-files-distribution\Generated_embeddings_"

for frechet in fad:
    # Compute FAD score while reusing the saved embeddings
    # (or saving new ones if paths are provided and embeddings don't exist yet)
    scores[0][frechet.model_name] = 0
    fad_score = frechet.score(
        path_to_background_set,
        path_to_eval_set,
        background_embds_path=background_embds_path + frechet.model_name + ".npy",
        eval_embds_path=eval_embds_path + frechet.model_name + ".npy",
        dtype="float32"
    )
    print(frechet.model_name, ": ", fad_score)
    scores[0][frechet.model_name] = fad_score

np.save("scores.npy", scores)
x = "stop here"
