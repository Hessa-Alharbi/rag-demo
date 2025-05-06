"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import Link from "next/link"
import Image from "next/image"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"
import { motion, AnimatePresence } from "framer-motion"
import apiClient from "@/lib/api-client"

import { Button } from "@/components/ui/button"
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { useAuth } from "@/lib/auth-context"

const loginSchema = z.object({
  username_or_email: z.string().min(1, "Username or email is required"),
  password: z.string().min(1, "Password is required"),
})

type LoginValues = z.infer<typeof loginSchema>

// Floating particles animation
const FloatingParticles = () => {
  const particles = Array.from({ length: 50 }).map((_, i) => ({
    id: i,
    size: Math.random() * 4 + 2,
    x: Math.random() * 100,
    y: Math.random() * 100,
    duration: Math.random() * 20 + 10
  }));

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {particles.map((particle) => (
        <motion.div
          key={particle.id}
          className="absolute rounded-full bg-primary/10"
          style={{
            width: particle.size,
            height: particle.size,
            left: `${particle.x}%`,
            top: `${particle.y}%`,
          }}
          animate={{
            y: [0, -20, 0],
            x: [0, Math.random() * 20 - 10, 0],
            opacity: [0.4, 0.8, 0.4]
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      ))}
    </div>
  );
};

// 3D Card hover effect
const Card3D = ({ children }: { children: React.ReactNode }) => {
  return (
    <motion.div
      className="w-full perspective-1000"
      whileHover={{ scale: 1.02 }}
      transition={{ type: 'spring', stiffness: 300, damping: 20 }}
    >
      {children}
    </motion.div>
  );
};

// Logo animation
const AnimatedLogo = ({ size = 180 }: { size?: number }) => {
  return (
    <motion.div
      initial={{ scale: 0, rotate: -10 }}
      animate={{ scale: 1, rotate: 0 }}
      transition={{
        type: "spring",
        stiffness: 260,
        damping: 20,
        delay: 0.3
      }}
      whileHover={{ 
        rotate: [0, -5, 5, -5, 0],
        transition: { duration: 0.5 }
      }}
    >
      <Image
        src="/Transparent-Navid-Logo.png"
        alt="Navid Logo"
        width={size}
        height={size}
        className="mx-auto"
      />
    </motion.div>
  );
};

export default function LoginPage() {
  const router = useRouter()
  const { login } = useAuth()
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(false)
  const [mounted, setMounted] = useState(false)
  const [returnTo, setReturnTo] = useState<string | null>(null)

  // For ensuring animations run after mount
  useEffect(() => {
    setMounted(true)
    
    // Check if we have a returnTo parameter in the URL
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search)
      const returnToParam = params.get('returnTo')
      if (returnToParam) {
        setReturnTo(returnToParam)
        console.log('Found returnTo URL:', returnToParam)
      }
      
      // Check if user was just registered
      const registered = params.get('registered') === 'true'
      if (registered) {
        // Show success message for registration
        setError("Registration successful! Please log in with your credentials.")
      }
    }
  }, [])

  const form = useForm<LoginValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      username_or_email: "",
      password: "",
    },
  })

  const onSubmit = async (data: LoginValues) => {
    setIsLoading(true)
    setError(null)
    
    try {
      // Use the auth context to handle login
      await login(data.username_or_email, data.password)
      // If we reach here, login was successful and the auth context will handle redirection
    } catch (err: any) {
      console.error("Login error:", err)
      // Display more user-friendly error messages
      if (err.response?.status === 401) {
        setError("Invalid username or password. Please try again.")
      } else if (err.response?.status === 404) {
        setError("User not found. Please check your credentials.")
      } else if (err.message) {
        setError(err.message)
      } else {
        setError("Login failed. Please check your internet connection and try again.")
      }
    } finally {
      setIsLoading(false)
    }
  }

  // Background elements animation
  const backgroundVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { duration: 1.5 }
    }
  };

  // Text animation variants
  const textVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: (custom: number) => ({
      opacity: 1,
      y: 0,
      transition: { 
        delay: custom * 0.2 + 0.5,
        duration: 0.8,
        type: "spring",
        stiffness: 100
      }
    })
  };

  // Form animation
  const formVariants = {
    hidden: { opacity: 0, scale: 0.9 },
    visible: { 
      opacity: 1, 
      scale: 1,
      transition: { 
        delay: 0.3,
        duration: 0.5,
        type: "spring",
        stiffness: 300,
        damping: 25
      }
    }
  };

  // Input field animation
  const inputVariants = {
    hidden: { x: -20, opacity: 0 },
    visible: (custom: number) => ({
      x: 0,
      opacity: 1,
      transition: { 
        delay: custom * 0.15 + 0.6,
        duration: 0.5,
        type: "spring"
      }
    }),
    focus: { 
      scale: 1.02,
      boxShadow: "0 0 8px rgba(var(--color-primary-rgb), 0.5)"
    }
  };

  if (!mounted) {
    return null; // Prevent flash of un-animated content
  }

  return (
    <div className="relative flex min-h-screen bg-gradient-to-br from-background to-background/80 overflow-hidden">
      <FloatingParticles />
      
      {/* Left side - Welcome Message and Logo */}
      <motion.div 
        className="hidden md:flex md:w-1/2 bg-primary/5 backdrop-blur-sm flex-col justify-center items-center p-10 relative z-10"
        initial="hidden"
        animate="visible"
        variants={backgroundVariants}
      >
        <div className="max-w-md text-center relative">
          <AnimatedLogo size={180} />
          
          <motion.h1 
            custom={1}
            variants={textVariants}
            initial="hidden"
            animate="visible"
            className="text-5xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-primary to-primary/80"
          >
            Welcome to Navid's RAG platform
          </motion.h1>
          
          <motion.p 
            custom={2}
            variants={textVariants}
            initial="hidden"
            animate="visible"
            className="text-xl mb-6"
          >
            The next-generation chat interface for Retrieval-Augmented Generation
          </motion.p>
          
          <motion.p 
            custom={3}
            variants={textVariants}
            initial="hidden"
            animate="visible"
            className="text-muted-foreground"
          >
            Sign in to your account to access your chats and settings.
          </motion.p>

          {/* Animated decorative elements */}
          <motion.div 
            className="absolute -bottom-16 -left-16 w-32 h-32 rounded-full bg-primary/5"
            animate={{ 
              scale: [1, 1.2, 1],
              opacity: [0.3, 0.6, 0.3],
            }}
            transition={{ 
              duration: 8,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
          
          <motion.div 
            className="absolute -top-20 -right-10 w-40 h-40 rounded-full bg-primary/5"
            animate={{ 
              scale: [1, 0.8, 1],
              opacity: [0.2, 0.5, 0.2],
            }}
            transition={{ 
              duration: 10,
              repeat: Infinity,
              ease: "easeInOut",
              repeatType: "reverse"
            }}
          />
        </div>
      </motion.div>

      {/* Right side - Login Form */}
      <div className="w-full md:w-1/2 flex items-center justify-center p-4 relative z-10">
        <motion.div 
          variants={formVariants}
          initial="hidden"
          animate="visible"
          className="w-full max-w-md"
        >
          <Card3D>
            <Card className="backdrop-blur-sm bg-card/90 border border-primary/10 shadow-lg">
              {/* Logo visible only on mobile */}
              <div className="md:hidden flex justify-center mt-6 mb-2">
                <AnimatedLogo size={100} />
              </div>
              
              <CardHeader className="space-y-1">
                <motion.div
                  initial={{ opacity: 0, y: -20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.4 }}
                >
                  <CardTitle className="text-2xl font-bold text-center">Login</CardTitle>
                  <CardDescription className="text-center">
                    Enter your credentials to access your account
                  </CardDescription>
                </motion.div>
              </CardHeader>
              
              <CardContent>
                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <Alert variant="destructive" className="mb-4">
                        <AlertDescription>{error}</AlertDescription>
                      </Alert>
                    </motion.div>
                  )}
                </AnimatePresence>
                
                <Form {...form}>
                  <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                    <motion.div
                      custom={1}
                      variants={inputVariants}
                      initial="hidden"
                      animate="visible"
                      whileFocus="focus"
                    >
                      <FormField
                        control={form.control}
                        name="username_or_email"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Username or Email</FormLabel>
                            <FormControl>
                              <Input 
                                placeholder="Enter your username or email" 
                                {...field} 
                                disabled={isLoading}
                                className="transition-all duration-300"
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </motion.div>
                    
                    <motion.div
                      custom={2}
                      variants={inputVariants}
                      initial="hidden"
                      animate="visible"
                    >
                      <FormField
                        control={form.control}
                        name="password"
                        render={({ field }) => (
                          <FormItem>
                            <FormLabel>Password</FormLabel>
                            <FormControl>
                              <Input 
                                type="password" 
                                placeholder="Enter your password" 
                                {...field} 
                                disabled={isLoading}
                                className="transition-all duration-300"
                              />
                            </FormControl>
                            <FormMessage />
                          </FormItem>
                        )}
                      />
                    </motion.div>
                    
                    <motion.div
                      custom={3}
                      variants={inputVariants}
                      initial="hidden"
                      animate="visible"
                      whileHover={{ scale: isLoading ? 1 : 1.03 }}
                      whileTap={{ scale: isLoading ? 1 : 0.98 }}
                    >
                      <Button 
                        type="submit" 
                        className="w-full relative overflow-hidden group" 
                        disabled={isLoading}
                      >
                        {isLoading ? (
                          <span className="flex items-center justify-center">
                            <span className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-b-transparent"></span>
                            Logging in...
                          </span>
                        ) : (
                          <>
                            <span className="relative z-10">Login</span>
                            <motion.span 
                              className="absolute inset-0 bg-primary-foreground/10"
                              initial={{ x: "-100%" }}
                              whileHover={{ x: "100%" }}
                              transition={{ duration: 0.5 }}
                            />
                          </>
                        )}
                      </Button>
                    </motion.div>
                  </form>
                </Form>
              </CardContent>
              
              <CardFooter className="flex flex-col space-y-2">
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.9, duration: 0.5 }}
                  className="text-center text-sm"
                >
                  <Link 
                    href="/auth/forgot-password" 
                    className="text-primary hover:underline relative inline-block"
                  >
                    Forgot your password?
                    <motion.span 
                      className="absolute bottom-0 left-0 w-0 h-0.5 bg-primary"
                      whileHover={{ width: "100%" }}
                      transition={{ duration: 0.3 }}
                    />
                  </Link>
                </motion.div>
                
                <motion.div 
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 1, duration: 0.5 }}
                  className="text-center text-sm"
                >
                  Don&apos;t have an account?{" "}
                  <Link 
                    href="/auth/register" 
                    className="text-primary hover:underline relative inline-block"
                  >
                    Register
                    <motion.span 
                      className="absolute bottom-0 left-0 w-0 h-0.5 bg-primary"
                      whileHover={{ width: "100%" }}
                      transition={{ duration: 0.3 }}
                    />
                  </Link>
                </motion.div>
              </CardFooter>
            </Card>
          </Card3D>
        </motion.div>
      </div>
    </div>
  )
} 