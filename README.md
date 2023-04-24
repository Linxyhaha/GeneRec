# GeneRec
This is the pytorch implementation of our paper:
> Generative Recommendation: Towards Next-generation Recommender Paradigm

## Environment
- Anaconda 3
- python 3.8.15
- pytorch 1.7.0
- numpy 1.23.5

## Usage

### AI Editor
The codes for AI Editor are in './code/AI_Editor/' folder, including three tasks 1) thumbnail selection & thumbnail generation, 2) micro-video clipping, and 3) micro-video content editing.

#### Thumbnail selection
The personalized selected thumbnails can be obtained by running
```
cd code/AI_Editor
python select_thumbnail.py
```

#### Thumbnail generation
The personalized generated thumbnails can be obtained by running
```
cd code/AI_Editor/Thumbnail_gen
python scripts/generate_thumbnail.py
```

#### Micro-video clipping
The personalized clipped micro-videos can be obtained by running
```
cd code/AI_Editor
python clip_microVid.py
```

#### Micro-video content editing
This task requires **training**, which can be achieved by running
```
cd code/AI_Editor/Content_editing/scripts
sh train.sh
```

At **inference** stage, the personalized edited micro-videos can be obtained by running
```
cd code/AI_Editor/Content_editing/scripts
sh edit.sh
```

### AI Creator
The codes for AI Creator are in './code/AI_Creator/' folder, including the task of micro-video content creation.

#### Micro-video content creation
This task also requires **training**, which can be achieved by running
```
cd code/AI_Creator/Content_creation/scripts
sh train.sh
```

At **inference** stage, the personalized created micro-videos can be obtained by running
```
cd code/AI_Creator/Content_creation/scripts
sh gen.sh
```

### Data
Unfortunately, due to the data privacy concerns, we are unable to release the experimental datasets.