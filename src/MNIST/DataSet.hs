module MNIST.DataSet where

import MNIST.Prelude

import qualified Codec.Compression.GZip  as GZip
import qualified Data.ByteString.Lazy    as L
import qualified Data.Vector.Generic     as G
import qualified Numeric.LinearAlgebra   as HM
import qualified Numeric.LinearAlgebra.Static  as SM
import qualified Data.Vector.Unboxed.Base as UV

rows :: Int
rows = 28

cols :: Int
cols = 28

nPixels :: Int
nPixels = cols * rows

nLabels :: Int
nLabels = 10


_loadMNIST
    :: FilePath
    -> FilePath
    -> IO [(Int, UVector Int)]
_loadMNIST dataPath labelPath = do
  runMaybeT $ do
    i <- MaybeT . fmap (decodeIDX       . GZip.decompress) . L.readFile $ dataPath
    l <- MaybeT . fmap (decodeIDXLabels . GZip.decompress) . L.readFile $ labelPath
    d <- MaybeT . pure $ labeledIntData l i
    return d
  >>= \case
    Just mnist -> return mnist
    Nothing    -> throwString $
      "couldn't read gzipped MNIST at data file " <> dataPath
      <> " and label file " <> labelPath

loadMNISTTf :: FilePath -> FilePath -> IO [(Int, Vector Int)]
loadMNISTTf dp lp = _loadMNIST dp lp
  >>= pure . fmap (identity *** G.convert)

trainingDataTf :: IO [(Int, Vector Int)]
trainingDataTf =
  loadMNISTTf
    "data/train-images-idx3-ubyte.gz"
    "data/train-labels-idx1-ubyte.gz"

testDataTf :: IO [(Int, Vector Int)]
testDataTf =
  loadMNISTTf
    "data/t10k-images-idx3-ubyte.gz"
    "data/t10k-labels-idx1-ubyte.gz"

-- ========================================================================= --

loadMNISTBp
    :: FilePath
    -> FilePath
    -> IO [(R 784, R 9)]
loadMNISTBp dp lp = _loadMNIST dp lp
  >>= pure . fmap ((fromJust . mkImage *** fromJust . mkLabel) . swap)

  where
    mkImage :: UVector Int -> Maybe (R 784)
    mkImage u = (SM.create . G.convert . G.map (\i -> fromIntegral i / 255) $ u)

    mkLabel :: Int -> Maybe (R 9)
    mkLabel n = SM.create $ HM.build 9 (fromIntegral . fromEnum . (== n) . round)


trainingDataBp :: IO [(R 784, R 9)]
trainingDataBp =
  loadMNISTBp
    "data/train-images-idx3-ubyte.gz"
    "data/train-labels-idx1-ubyte.gz"

testDataBp :: IO [(R 784, R 9)]
testDataBp =
  loadMNISTBp
    "data/t10k-images-idx3-ubyte.gz"
    "data/t10k-labels-idx1-ubyte.gz"



