class PauseButtonManager {
    constructor() {
        this.isPaused = false;
        this.animationId = null;
        this.lastTime = 0;
        this.accumulatedTime = 0;
        this.targetFPS = 60;
        this.frameInterval = 1000 / this.targetFPS;
    }

    pause() {
        if (this.isPaused) return;
        
        this.isPaused = true;
        this.accumulatedTime = 0;
        
        // 停止动画循环
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        console.log('动画已暂停');
    }

    resume() {
        if (!this.isPaused) return;
        
        this.isPaused = false;
        this.lastTime = performance.now();
        
        // 重新启动动画循环
        this.startAnimationLoop();
        
        console.log('动画已恢复');
    }

    startAnimationLoop() {
        if (this.isPaused) return;
        
        const loop = (currentTime) => {
            if (this.isPaused) return;
            
            const deltaTime = currentTime - this.lastTime;
            this.accumulatedTime += deltaTime;
            
            // 控制帧率
            if (this.accumulatedTime >= this.frameInterval) {
                this.accumulatedTime -= this.frameInterval;
                this.lastTime = currentTime;
                
                // 调用原始的动画函数
                if (window.originalAnimate) {
                    window.originalAnimate();
                }
            }
            
            this.animationId = requestAnimationFrame(loop);
        };
        
        this.animationId = requestAnimationFrame(loop);
    }

    toggle() {
        if (this.isPaused) {
            this.resume();
        } else {
            this.pause();
        }
    }

    isAnimationPaused() {
        return this.isPaused;
    }
}

// 等待 DOM 加载完成
document.addEventListener('DOMContentLoaded', () => {
    // 创建暂停管理器实例
    const pauseManager = new PauseButtonManager();
    
    // 暴露到全局作用域
    window.PauseButtonManager = pauseManager;
    
    // 保存原始的 animate 函数
    if (window.animate) {
        window.originalAnimate = window.animate;
        
        // 重写 animate 函数以支持暂停
        window.animate = function() {
            if (!pauseManager.isPaused) {
                window.originalAnimate();
            }
        };
    }
    
    console.log('暂停按钮管理器已初始化');
});

// 提供便捷的全局函数
window.togglePause = () => {
    if (window.PauseButtonManager) {
        window.PauseButtonManager.toggle();
    }
};

window.isPaused = () => {
    return window.PauseButtonManager ? window.PauseButtonManager.isAnimationPaused() : false;
};