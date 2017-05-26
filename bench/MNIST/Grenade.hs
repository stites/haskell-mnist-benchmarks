module Main where

import MNIST.Prelude
import Control.Monad.Random
import qualified Numeric.LinearAlgebra.Static as SA

import Grenade


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


randomNet :: MonadRandom m => m MNIST
randomNet = randomNetwork


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


netScore :: MNIST -> IO ()
netScore network = putStrLn . unlines $
  (fmap.fmap) (showNorm . go) testIns

  where
    testIns :: [[(Double, Double)]]
    testIns = [ [ (x,y)  | x <- [0..50] ] | y <- [0..20] ]

    go :: (Double, Double) -> Output
    go (x,y) = runNet network (S1D $ SA.vector [x / 25 - 1, y / 10 - 1])

    showNorm :: Output -> Char
    showNorm (S1D r) = render $ SA.mean r

    render :: Double -> Char
    render n | n <= 0.2  = ' '
             | n <= 0.4  = '.'
             | n <= 0.6  = '-'
             | n <= 0.8  = '='
             | otherwise = '#'


main :: IO ()
main = do
  net0 <- randomNet
  net  <- netTrain net0 params examples
  netScore net
  where
    examples :: Int
    examples = 10000

    params :: LearningParameters
    params = LearningParameters
      { learningRate = 0.01
      , learningMomentum = 0.9
      , learningRegulariser = 0.0005
      }

