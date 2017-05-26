module Main where

-- import qualified Data.Attoparsec.Text as A
import MNIST.Prelude
import Data.List ( foldl' )
import Data.Semigroup ( (<>) )
import qualified Data.Text as T
import qualified Data.Text.IO as T
import qualified Data.Vector.Storable as V
import Control.Monad.Random

import Numeric.LinearAlgebra ( maxIndex )
import qualified Numeric.LinearAlgebra.Static as SA

import Grenade
import Grenade.Utils.OneHot


type MNIST
  = Network
    '[ FullyConnected 784 300, Logit
     , FullyConnected 300 100, Logit
     , FullyConnected 100   9, Logit
     ]
    '[ 'D1 784, 'D1 300
     , 'D1 300, 'D1 100
     , 'D1 100, 'D1 9
     , 'D1 9
     ]


randomMnist :: MonadRandom m => m MNIST
randomMnist = randomNetwork


type Input  = S ('D1 784)
type Output = S ('D1   9)
type DataSet = [(Input, Output)]


trainAll :: LearningParameters -> MNIST -> DataSet -> MNIST
trainAll rate net0 dataset = foldl' trainOne net0 dataset
  where
  trainOne :: MNIST -> (Input, Output) -> MNIST
  trainOne !network (i,o) = train rate network i o


netTrain :: MNIST -> LearningParameters -> Int -> IO MNIST
netTrain net0 rate n = do
  inps <- replicateM n randomTrainingData
  let outs = map randomTrainingLabels inps
  return $ trainAll rate net0 (zip inps outs)
  where
    randomTrainingData :: IO Input
    randomTrainingData = do
      s <- getRandom
      return . S1D $ SA.randomVector s SA.Uniform * 2 - 1

    randomTrainingLabels :: Input -> Output
    randomTrainingLabels (S1D v) = S1D . fromIntegral $ fromEnum isTrue
      where
        isTrue :: Bool
        isTrue = (v `inCircle` (fromRational    0.33, 0.33))
              || (v `inCircle` (fromRational (-0.33), 0.33))

    inCircle :: KnownNat n => R n -> (R n, Double) -> Bool
    inCircle v (o, r) = SA.norm_2 (v - o) <= r

main = undefined

