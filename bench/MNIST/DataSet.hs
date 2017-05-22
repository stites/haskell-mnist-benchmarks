module MNIST.DataSet where

import MNIST.Prelude

import qualified Codec.Compression.GZip  as GZip
import qualified Data.ByteString.Lazy    as L
import qualified Data.Text               as Text
import qualified Data.Vector.Unboxed     as V


rows :: Int
rows = 28


cols :: Int
cols = 28


readMNIST :: FilePath -> FilePath -> IO [(Int, UVector Int)]
readMNIST dataPath labelPath = do
  runMaybeT $ do
    d  <- MaybeT . fmap (decodeIDX       . GZip.decompress) . L.readFile $ dataPath
    ls <- MaybeT . fmap (decodeIDXLabels . GZip.decompress) . L.readFile $ labelPath
    MaybeT . pure $ labeledIntData ls d
  >>= \case
    Just mnist -> return mnist
    Nothing    -> throwString $
      "couldn't read gzipped MNIST at data file " <> dataPath <> " and label file " <> labelPath


trainingData :: IO [(Int, UVector Int)]
trainingData = readMNIST "data/train-images-idx3-ubyte.gz" "data/train-labels-idx1-ubyte.gz"

testData :: IO [(Int, UVector Int)]
testData = readMNIST "data/t10k-images-idx3-ubyte.gz" "data/t10k-labels-idx1-ubyte.gz"

