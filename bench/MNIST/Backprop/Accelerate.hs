{-# OPTIONS_GHC -fno-warn-orphans #-}
module Main where

import MNIST.Prelude
import MNIST.DataSet
import Numeric.LinearAlgebra.Static hiding (dot)
import Numeric.Backprop

import qualified Data.Vector.Generic             as VG
import qualified Data.Vector                     as V
import qualified Generics.SOP                    as SOP
import qualified Numeric.LinearAlgebra           as HM
import qualified System.Random.MWC               as MWC
import qualified System.Random.MWC.Distributions as MWC

-- | type-combinators alias for the terminal constructor
nil :: Prod f '[]
nil = Ø

-- ========================================================================= --
-- Internal structure of our FF neural net

data Layer i o = Layer
  { _lWeights :: !(L o i)
  , _lBiases  :: !(R o)
  } deriving (Show, Generic)

instance SOP.Generic (Layer i o)
instance NFData (Layer i o)

data Network i h1 h2 o = Net
  { _nLayer1 :: !(Layer i  h1)
  , _nLayer2 :: !(Layer h1 h2)
  , _nLayer3 :: !(Layer h2 o)
  } deriving (Show, Generic)

instance SOP.Generic (Network i h1 h2 o)
instance NFData (Network i h1 h2 o)

-- ========================================================================= --
-- type classes

type KnownNat4 i h1 h2 o = (KnownNat i, KnownNat h1, KnownNat h2, KnownNat o)
type KnownNat2 i o = (KnownNat i, KnownNat o)


instance KnownNat2 i o => Num (Layer i o) where
  Layer w1 b1 + Layer w2 b2 = Layer (w1 + w2) (b1 + b2)
  Layer w1 b1 - Layer w2 b2 = Layer (w1 - w2) (b1 - b2)
  Layer w1 b1 * Layer w2 b2 = Layer (w1 * w2) (b1 * b2)
  abs    (Layer w b)        = Layer (abs    w) (abs    b)
  signum (Layer w b)        = Layer (signum w) (signum b)
  negate (Layer w b)        = Layer (negate w) (negate b)
  fromInteger x = Layer (fromInteger x) (fromInteger x)

instance KnownNat4 i h1 h2 o => Num (Network i h1 h2 o) where
  Net a b c + Net d e f = Net (a + d) (b + e) (c + f)
  Net a b c - Net d e f = Net (a - d) (b - e) (c - f)
  Net a b c * Net d e f = Net (a * d) (b * e) (c * f)
  abs    (Net a b c)    = Net (abs    a) (abs    b) (abs    c)
  signum (Net a b c)    = Net (signum a) (signum b) (signum c)
  negate (Net a b c)    = Net (negate a) (negate b) (negate c)
  fromInteger x         = Net (fromInteger x) (fromInteger x) (fromInteger x)

instance KnownNat2 i o => Fractional (Layer i o) where
  Layer w1 b1 / Layer w2 b2 = Layer (w1 / w2) (b1 / b2)
  recip (Layer w b)         = Layer (recip w) (recip b)
  fromRational x            = Layer (fromRational x) (fromRational x)

instance KnownNat4 i h1 h2 o => Fractional (Network i h1 h2 o) where
  Net a b c / Net d e f = Net (a / d) (b / e) (c / f)
  recip (Net a b c)     = Net (recip a) (recip b) (recip c)
  fromRational x        = Net (fromRational x) (fromRational x) (fromRational x)

instance KnownNat n => MWC.Variate (R n) where
  uniform g = randomVector <$> MWC.uniform g <*> pure Uniform
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance KnownNat2 m n => MWC.Variate (L m n) where
  uniform g = uniformSample <$> MWC.uniform g <*> pure 0 <*> pure 1
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance KnownNat2 i o => MWC.Variate (Layer i o) where
  uniform g = Layer <$> MWC.uniform g <*> MWC.uniform g
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g

instance KnownNat4 i h1 h2 o => MWC.Variate (Network i h1 h2 o) where
  uniform g = Net <$> MWC.uniform g <*> MWC.uniform g <*> MWC.uniform g
  uniformR (l, h) g = (\x -> x * (h - l) + l) <$> MWC.uniform g


-- ========================================================================= --
-- Basic math functions with back propagation

matVec :: forall m n . (KnownNat m, KnownNat n) => Op '[ L m n, R n ] (R m)
matVec = op2' $ \m v -> (forward m v, backward m v)
  where
    forward :: L m n -> R n -> R m
    forward m v = m #> v

    backward :: L m n -> R n -> Maybe (R m) -> (L m n, R n)
    backward m v (fromMaybe 1 -> g) = (g `outer` v, tr m #> g)


dot :: forall n . KnownNat n => Op '[ R n, R n ] Double
dot = op2' $ \x y -> (forward x y, backward x y)
  where
    forward :: R n -> R n -> Double
    forward x y = x <.> y

    backward :: R n -> R n -> Maybe Double -> (R n, R n)
    backward x y = \case
      Nothing -> (y, x)
      Just g  -> (konst g * y, x * konst g)


scale :: forall n . KnownNat n => Op '[ Double, R n ] (R n)
scale = op2' $ \a x -> (forward a x, backward a x)
  where
    forward :: Double -> R n -> R n
    forward a x = konst a * x

    backward :: Double -> R n -> Maybe (R n) -> (Double, R n)
    backward a x = \case
      Nothing -> (HM.sumElements (extract x      ), konst a    )
      Just g  -> (HM.sumElements (extract (x * g)), konst a * g)


vsum :: forall n . KnownNat n => Op '[ R n ] Double
vsum = op1' $ \x -> (forward x, backward)
  where
    forward :: R n -> Double
    forward = HM.sumElements . extract

    backward :: Maybe Double -> R n
    backward = maybe 1 konst


logistic :: Floating a => a -> a
logistic x = 1 / (1 + exp (-x))


-- ========================================================================= --
-- run backpropagation
type LayerCtx i o = '[R i, Layer i o]
type NetCtx i h1 h2 o = '[ R i, Network i h1 h2 o ]

runLayer :: forall i o s . (KnownNat i, KnownNat o) => BPOp s '[ R i, Layer i o ] (R o)
runLayer = withInps $ \(decombinate -> (x, l)) -> do
  w :< b :< _ <- partsVar gTuple l
  y <- opVar matVec (w :< x :< nil)
  return $ y + b
  where
    decombinate
      :: Prod (BVar s (LayerCtx i o)) '[R i, Layer i o]
      -> (BVar s (LayerCtx i o) (R i), BVar s (LayerCtx i o) (Layer i o))
    decombinate (x :< l :< _) = (x, l)


runNetwork :: KnownNat4 i h1 h2 o => BPOp s (NetCtx i h1 h2 o) (R o)
runNetwork = withInps $ \(x :< n :< _) -> do
  l1 :< l2 :< l3 :< _ <- partsVar gTuple n

  y <- bindVar $ liftB (bpOp runLayer) (x :< l1 :< nil)
  -- or
  z <- bindVar $ (bpOp runLayer) .$ (logistic y :< l2 :< nil)
  -- or
  r <- (bpOp runLayer) ~$ (logistic z :< l3 :< nil)

  bpOp softmax ~$ only r
  where
    softmax :: KnownNat n => BPOp s '[ R n ] (R n)
    softmax = withInps $ \(x :< _) -> do
      expX <- bindVar (exp x)
      totX <- vsum ~$ (expX :< nil)
      scale        ~$ (1 / totX :< expX :< nil)


runNetOnInp :: KnownNat4 i h1 h2 o => Network i h1 h2 o -> R i -> R o
runNetOnInp n x = evalBPOp runNetwork (x ::< n ::< nil)


gradNet :: KnownNat4 i h1 h2 o => Network i h1 h2 o -> R i -> Network i h1 h2 o
gradNet n x = case gradBPOp runNetwork (x ::< n ::< nil) of
    _gradX ::< gradN ::< nil -> gradN


-- ========================================================================= --

-- crossEntropy :: forall s n . KnownNat n => R n -> BPOp s '[ R n ] Double
-- crossEntropy targ = withInps $ \(r :< _) ->
--   negate (dot ~$ (log r :< only t))
--   where
--     t :: BVar s '[R n] (R n)
--     t = constVar targ


crossEntropyI :: forall s n . KnownNat n => R n -> BPOpI s '[ R n ] Double
crossEntropyI targ (r :< _) = negate (dot .$ (log r :< only t))
  where
    t :: BVar s '[R n] (R n)
    t = constVar targ


softMaxCrossEntropy :: forall s n . KnownNat n => R n -> BPOp s '[ R n ] Double
softMaxCrossEntropy targ = withInps $ \(r :< Ø) -> do
  bindVar $ realToFrac tsum * log (vsum .$ (only r)) - (dot .$ (r :< t :< nil))
  where
    tsum :: Double
    tsum = HM.sumElements . extract $ targ

    t :: BVar s '[R n] (R n)
    t = constVar targ


softMaxCrossEntropyI :: forall s n . KnownNat n => R n -> BPOpI s '[ R n ] Double
softMaxCrossEntropyI targ (r :< Ø) =
  realToFrac tsum * log (vsum .$ (only r)) - (dot .$ (r :< t :< nil))
  where
    tsum :: Double
    tsum = HM.sumElements . extract $ targ

    t :: BVar s '[R n] (R n)
    t = constVar targ


trainStep
  :: forall i h1 h2 o. KnownNat4 i h1 h2 o
  => Double
  -> R i
  -> R o
  -> Network i h1 h2 o
  -> Network i h1 h2 o
trainStep r !x !t !n =
  case gradBPOp o (x ::< n ::< nil) of
    (_ :< I gN :< _) -> n - (realToFrac r * gN)
  where
    o :: BPOp s '[ R i, Network i h1 h2 o ] Double
    o = do
      y <- runNetwork
      implicitly (crossEntropyI t) -$ (y :< nil)

trainList
  :: KnownNat4 i h1 h2 o
  => Double
  -> [(R i, R o)]
  -> Network i h1 h2 o
  -> Network i h1 h2 o
trainList r = flip $ foldl' (\n (x,y) -> trainStep r x y n)

testNet
  :: forall i h1 h2 o. KnownNat4 i h1 h2 o
  => [(R i, R o)]
  -> Network i h1 h2 o
  -> Double
testNet xs n = sum (map (\(i,o) -> test i o) xs) / fromIntegral (length xs)
  where
    test :: R i -> R o -> Double
    test x (extract->t) = fromIntegral . fromEnum $
      HM.maxIndex t == HM.maxIndex (extract r)
      where
        r :: R o
        r = evalBPOp runNetwork (x ::< n ::< nil)


main :: IO ()
main = MWC.withSystemRandom $ \g -> do
  -- initialize data and network
  !trainingSet <- trainingDataBp
  !testSet     <- testDataBp
  !net0        <- MWC.uniformR @(Network 784 300 100 9) (-0.5, 0.5) g

  flip evalStateT net0 . forM_ [1..100] $ \e -> do
    trainingSet' <- liftIO . fmap V.toList $ MWC.uniformShuffle (V.fromList trainingSet) g
    liftIO $ printf "[Epoch %d]\n" (e :: Int)

    forM_ ([1..] `zip` chunksOf batch trainingSet') $ \(b, chnk) -> StateT $ \n0 -> do
      printf "(Batch %d)\n" (b :: Int)

      -- t0 <- getCurrentTime
      n' <- evaluate . force $ trainList rate chnk n0
      -- t1 <- getCurrentTime
      -- printf "Trained on %d points in %s.\n" batch (show (t1 `diffUTCTime` t0))

      let trainScore = testNet chnk    n'
          testScore  = testNet testSet n'
      printf "Training error:   %.2f%%\n" ((1 - trainScore) * 100)
      -- printf "Validation error: %.2f%%\n" ((1 - testScore ) * 100)

      return ((), n')
  where
    rate = 0.0001
    batch = 100

  --  go :: StateT (Network 784 300 100 9) IO ()
  --  go = do
  --    e <- get

--
--   -- Test
--   let testPreds = map (take 3 testSet) $ \(x, _) -> evalBPOp runNetwork (x ::< net0 ::< nil)
--
--   liftIO $ forM_ ([0..3] :: [Int]) $ \i -> do
--       -- T.putStrLn $ drawMNIST $ testImages !! i
--       putStrLn $ "\n" ++ "expected " ++ show (testLabels !! i)
--       putStrLn $         "     got " ++ show (testPreds !! i)
--
--
